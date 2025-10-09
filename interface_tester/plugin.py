# Copyright 2022 Canonical Ltd.
# See LICENSE file for licensing details.
import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type

from ops.testing import CharmType
from scenario import State
from scenario.errors import MetadataNotFoundError
from scenario.state import _CharmSpec

from interface_tester.collector import InterfaceTestSpec, gather_test_spec_for_version
from interface_tester.errors import (
    InterfaceTesterValidationError,
    InterfaceTestsFailed,
    NoTestsRun,
)
from interface_tester.interface_test import (
    RoleLiteral,
    _InterfaceTestContext,
    tester_context,
)
from interface_tester.schema_base import DataBagSchema

ROLE_TO_ROLE_META = {"provider": "provides", "requirer": "requires"}

logger = logging.getLogger("pytest_interface_tester")


class InterfaceTester:
    _RAISE_IMMEDIATELY = False

    def __init__(
        self,
        repo: str = "https://github.com/canonical/charm-relation-interfaces",
        branch: str = "main",
        base_path: str = "interfaces",
    ):
        self._repo = repo
        self._branch = branch
        self._base_path = base_path

        # set by .configure()
        self._charm_type = None
        self._meta = None
        self._actions = None
        self._config = None
        self._endpoint = None
        self._interface_name = None
        self._interface_version = 0
        self._juju_version = None
        self._state_template = None
        self._interface_subdir = ""
        self._tests_dir = "interface_tests"

        self._charm_spec_cache = None

    def configure(
        self,
        *,
        charm_type: Optional[Type[CharmType]] = None,
        repo: Optional[str] = None,
        branch: Optional[str] = None,
        base_path: Optional[str] = None,
        interface_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        interface_version: Optional[int] = None,
        state_template: Optional[State] = None,
        juju_version: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        actions: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        interface_subdir: Optional[str] = None,
        tests_dir: Optional[str] = None,
    ):
        """

        :arg interface_name: the interface to test.
        :arg endpoint: the endpoint to test.
            If omitted, will test all endpoints with this interface.
        :param interface_version: what version of this interface we should be testing.
        :arg state_template: template state to use with the scenario test.
            The plugin will inject the relation spec under test, unless already defined.
        :param charm_type: The charm to test.
        :param repo: Repo to fetch the tests from.
        :param branch: Branch to fetch the tests from.
        :param base_path: path to an interfaces-compliant subtree within the repo.
        :param meta: charm metadata.yaml contents.
        :param actions: charm actions.yaml contents.
        :param config: charm config.yaml contents.
        :param juju_version: juju version that Scenario will simulate (also sets JUJU_VERSION
            envvar at charm runtime.)
        :param interface_subdir: Subdirectory to look for versioned interface directories in
            under interfaces/interface_name.
        :param tests_dir: Name of tests directory under
            interfaces/<interface_name>/<interface_subdir>/v<N>.
        """
        if charm_type:
            self._charm_type = charm_type
        if actions:
            self._actions = actions
        if meta:
            self._meta = meta
        if config:
            self._config = config
        if repo:
            self._repo = repo
        if endpoint:
            self._endpoint = endpoint
        if interface_name:
            self._interface_name = interface_name
        if interface_version is not None:
            self._interface_version = interface_version
        if state_template:
            self._state_template = state_template
        if branch:
            self._branch = branch
        if base_path:
            self._base_path = base_path
        if juju_version:
            self._juju_version = juju_version
        if interface_subdir is not None:
            self._interface_subdir = interface_subdir
        if tests_dir is not None:
            self._tests_dir = tests_dir

    def _validate_config(self):
        """Validate the configuration of the tester.

        Will raise InterfaceTesterValidationError if something is not right with the config.
        """
        errors = []
        if (self._actions or self._config) and not self._meta:
            errors.append(
                "Tester misconfigured: cannot set actions and config without setting meta."
            )
        if not self._charm_type:
            errors.append("Tester misconfigured: needs a charm_type set.")
        elif not self.meta:
            errors.append("no metadata: it was not provided, and it cannot be autoloaded")
        if not self._repo:
            errors.append("repo missing")
        if not self._interface_name:
            errors.append("interface_name missing")
        if not isinstance(self._interface_version, int):
            errors.append("interface_version should be an integer")
        if self._state_template and not isinstance(self._state_template, State):
            errors.append(
                f"state_template should be of type State, " f"not: {type(self._state_template)}"
            )
        if errors:
            err = "\n".join(errors)
            raise InterfaceTesterValidationError(
                f"pytest-interface-tester is misconfigured:\n{err}\n"
                f"please use the configure() API to provide the missing pieces."
            )

    @property
    def _charm_spec(self) -> _CharmSpec:
        """Return the _CharmSpec object representing the tested charm and its metadata."""
        if not self._charm_spec_cache:
            # We try to use Scenario's internal autoload functionality to autoload the charm spec.
            try:
                spec = _CharmSpec.autoload(self._charm_type)
                # if no metadata.yaml can be found in the charm type module's parent directory,
                # this call will raise:
            except MetadataNotFoundError as e:
                # if we have _meta set, we're good, otherwise we're misconfigured.
                if self._meta and self._charm_type:
                    spec = _CharmSpec(
                        meta=self._meta,
                        actions=self._actions,
                        config=self._config,
                        charm_type=self._charm_type,
                    )
                else:
                    raise InterfaceTesterValidationError(
                        "This InterfaceTester is missing charm metadata `meta` or a `charm type`. "
                        "Unable to load charm spec. Please provide both using the `configure` API."
                    ) from e
            self._charm_spec_cache = spec
        return self._charm_spec_cache

    @property
    def meta(self) -> dict:
        """Contents of the metadata.yaml of the charm being tested."""
        return self._meta or self._charm_spec.meta

    @property
    def actions(self) -> dict:
        """Contents of the actions.yaml of the charm being tested, if any."""
        return self._actions or self._charm_spec.actions

    @property
    def config(self) -> dict:
        """Contents of the config.yaml of the charm being tested, if any."""
        return self._config or self._charm_spec.config

    def _collect_interface_test_specs(self) -> InterfaceTestSpec:
        """Gathers the test cases as defined by charm-relation-interfaces, for both roles."""
        with tempfile.TemporaryDirectory() as tempdir:
            cmd = f"git clone --depth 1 --branch {self._branch} {self._repo}".split(" ")
            proc = Popen(cmd, cwd=tempdir, stderr=PIPE, stdout=PIPE)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"failed to fetch {self._repo}:{self._branch}, "
                    f"check that the ref is correct. "
                    f"out={proc.stdout.read()}"
                    f"err={proc.stderr.read()}"
                )

            repo_name = self._repo.split("/")[-1]
            if repo_name.endswith(".git"):
                repo_name = repo_name.rsplit(".", maxsplit=1)[0]

            intf_spec_path = (
                Path(tempdir)
                / repo_name
                / self._base_path
                / self._interface_name.replace("-", "_")
                / self._interface_subdir
                / f"v{self._interface_version}"
            )
            if not intf_spec_path.exists():
                raise RuntimeError(
                    f"interface spec dir not found at expected location. "
                    f"Check that {intf_spec_path} is a valid path in the remote repo you've "
                    f"selected as test case source for this run."
                )

            tests = gather_test_spec_for_version(
                intf_spec_path,
                interface_name=self._interface_name,
                version=self._interface_version,
                tests_dir=self._tests_dir,
            )

        return tests

    def _gather_supported_endpoints(self) -> Dict[RoleLiteral, List[str]]:
        """Given a relation interface name, return a list of supported endpoints as either role.

        These are collected from the charm's metadata.yaml.
        """
        supported_endpoints: Dict["RoleLiteral", List[str]] = {}
        role: RoleLiteral
        for role in ("provider", "requirer"):
            meta_role = ROLE_TO_ROLE_META[role]

            # assuming there's been a _validate_config() before this point, it's safe to access
            # `meta`.
            endpoints = self.meta.get(meta_role, {})
            # if there are no endpoints using this interface, this means that this charm does not
            # support that role in the relation. There MIGHT still be tests for the other role, but
            # they are then meant for a charm implementing the other role.

            endpoints_for_interface = [
                k for k, v in endpoints.items() if v["interface"] == self._interface_name
            ]

            if endpoints_for_interface:
                supported_endpoints[role] = endpoints_for_interface
            else:
                logger.warning(f"skipping role {role}: unsupported by this charm.")

        return supported_endpoints

    def _yield_tests(
        self,
    ) -> Generator[Tuple[Callable, RoleLiteral, DataBagSchema], None, None]:
        """Yield all test cases applicable to this charm and interface.

        This means:
        - collecting the test cases (InterfaceTestCase objects) as defined by the
          charm-relation-interfaces specification. These tests encode what it means to satisfy this
          relation interface, and include some optional set-up logic for the State the test has to
          be run with.
        - obtaining the mocker/charm spec, as provided by the charm repo which hosts the source of
          the charm we are currently testing.
        - obtain from the charm's metadata.yaml the endpoints supporting this interface (in either
          role).
        - for each endpoint, for each applicable test case, yield: the test case, the schema as
          specified by the interface, the event and the State.
        """

        interface_name = self._interface_name
        tests = self._collect_interface_test_specs()

        if not (tests["provider"]["tests"] or tests["requirer"]["tests"]):
            yield from ()
            return

        supported_endpoints = self._gather_supported_endpoints()
        if not supported_endpoints:
            raise RuntimeError(f"this charm does not declare any endpoint using {interface_name}.")

        role: RoleLiteral
        for role, endpoints in supported_endpoints.items():
            logger.debug(f"collecting scenes for {role}")

            spec = tests[role]
            schema = spec["schema"]
            for test in spec["tests"]:
                for endpoint in endpoints:
                    if self._endpoint and endpoint != self._endpoint:
                        logger.debug(f"skipped compatible endpoint {endpoint}")
                        continue
                    yield test, role, schema, endpoint

    def __repr__(self):
        return f"""<Interface Tester: 
        \trepo={self._repo}
        \tbranch={self._branch}
        \tbase_path={self._base_path}
        \tcharm_type={self._charm_type}
        \tmeta={self._meta}
        \tactions={self._actions}
        \tconfig={self._config}
        \tinterface_name={self._interface_name}
        \tinterface_version={self._interface_version}
        \tjuju_version={self._juju_version}
        \tstate_template={self._state_template}>"""

    @contextmanager
    def context(self, test_fn: Callable, role: RoleLiteral, schema: DataBagSchema, endpoint: str):
        logger.debug(f"Entering context {self!r}, {role=!r} {endpoint=!r} {schema=}.")
        self._validate_config()  # will raise if misconfigured
        ctx = _InterfaceTestContext(
            role=role,
            schema=schema,
            interface_name=self._interface_name,
            endpoint=endpoint,
            version=self._interface_version,
            charm_type=self._charm_type,
            state_template=self._state_template,
            meta=self.meta,
            config=self.config,
            actions=self.actions,
            supported_endpoints=self._gather_supported_endpoints(),
            test_fn=test_fn,
            juju_version=self._juju_version,
        )
        with tester_context(ctx):
            yield ctx

    def run(self) -> bool:
        """Run interface tests.

        Returns True if some tests were found and ran, False otherwise.
        """
        self._validate_config()  # will raise if misconfigured
        logger.info(f"Running {repr(self)}.")
        errors = []
        ran_some = False

        for test_fn, role, schema, endpoint in self._yield_tests():
            try:
                with self.context(test_fn, role, schema, endpoint) as ctx:
                    test_fn()
            except Exception as e:
                logger.exception(f"Interface tester plugin failed with {e}")

                if self._RAISE_IMMEDIATELY:
                    raise e

                errors.append((ctx, e))
            ran_some = True

        # todo: consider raising custom exceptions here.
        if errors:
            msgs = []
            for ctx, e in errors:
                msgs.append(
                    f" - {ctx.interface_name}[v{ctx.version}]@{ctx.role}:{ctx.test_fn} raised {e}"
                )
            long_msg = "\n".join(msgs)

            raise InterfaceTestsFailed(
                f"interface tests completed with {len(errors)} errors. \n" + long_msg
            )

        if not ran_some:
            msg = f"no tests gathered for {self._interface_name!r}/v{self._interface_version}"
            if self._endpoint:
                msg += f" and endpoint {self._endpoint!r}"
            logger.warning(msg)
            raise NoTestsRun(msg)
