# Copyright 2022 Canonical Ltd.
# See LICENSE file for licensing details.
"""
This module contains logic to gather interface tests from the relation interface specifications.
It also contains a `pprint_tests` function to display a pretty-printed listing of the
collected tests. This file is executable and will run that function when invoked.

If you are contributing a relation interface specification or modifying the tests, charms, or
schemas for one, you can execute this file to ascertain that all relevant data is being gathered
correctly.
"""

import dataclasses
import importlib
import inspect
import json
import logging
import sys
import types
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Type, TypedDict

import pydantic
import yaml

from interface_tester.interface_test import Role
from interface_tester.schema_base import DataBagSchema

logger = logging.getLogger("interface_tests_checker")

_DEFAULT_TESTS_DIR = "interface_tests"
_NotFound = object()


class _TestSetup(TypedDict):
    """Charm-specific configuration for the interface tester.

    Contains information to configure the tester."""

    charm_root: Optional[str]
    """Relative path, from the root of the repo, to where the root of the charm is.
    Useful for multi-charm monorepos. If not provided defaults to the charm name."""

    location: Optional[str]
    """Path to a python file, relative to the charm's git repo root, where the `identifier` 
    below can be found. If not provided defaults to "tests/interfaces/conftest.py" """

    identifier: Optional[str]
    """Name of a python identifier pointing to a pytest fixture yielding a 
    configured InterfaceTester instance. If not provided defaults to "interface_tester" """

    pre_run: Optional[str]
    """Bash script to do whatever it takes in this specific repo to generate a requirements.txt 
    file we can use to set up the venv to run the tests in. Useful for charms that use uv, poetry 
    or something else to manage dependencies."""


@dataclasses.dataclass
class _CharmTestConfig:
    name: str
    """The name of the charm."""
    url: str
    """Url of a git repository where the charm source can be found."""
    test_setup: Optional[_TestSetup] = None
    """Interface tester configuration. Can be left empty. All values will be defaulted."""
    branch: Optional[str] = None
    """Name of the git branch where to find the interface tester configuration. 
    If not provided defaults to "main". """

    def __hash__(self):
        return hash((self.name, self.url, self.branch))


class _InterfacesDotYamlSpec(TypedDict):
    """Specification of the `interface.yaml` file each interface/version dir should contain."""

    providers: List[_CharmTestConfig]
    requirers: List[_CharmTestConfig]
    maintainer: str


class _RoleTestSpec(TypedDict):
    """The tests, schema, and charms for a single role of a given relation interface version."""

    tests: List[Callable[[None], None]]
    schema: Optional[Type[DataBagSchema]]
    charms: List[_CharmTestConfig]


class InterfaceTestSpec(TypedDict):
    """The tests, schema, and charms for both roles of a given relation interface version."""

    provider: _RoleTestSpec
    requirer: _RoleTestSpec
    maintainer: str


def get_schema_from_module(module: object, name: str) -> Type[pydantic.BaseModel]:
    """Tries to get ``name`` from ``module``, expecting to find a pydantic.BaseModel."""
    schema_cls = getattr(module, name, None)
    if not schema_cls:
        raise NameError(name)
    if not issubclass(schema_cls, pydantic.BaseModel):
        raise TypeError(type(schema_cls))
    return schema_cls


def load_schema_module(schema_path: Path) -> types.ModuleType:
    """Import the schema.py file as a python module."""
    # so we can import without tricks
    sys.path.append(str(schema_path.parent))

    # strip .py
    module_name = str(schema_path.with_suffix("").name)

    # if a previous call to load_schema_module has loaded a
    # module with the same name, this will conflict.
    if module_name in sys.modules:
        del sys.modules[module_name]

    if pydantic.version.VERSION.split(".") <= ["2"]:
        # in pydantic v1 it's necessary; in v2 it isn't.

        # Otherwise we'll get an error when we re-run @validator
        logger.debug("Clearing pydantic.class_validators._FUNCS")
        pydantic.class_validators._FUNCS.clear()  # noqa

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise
    finally:
        # cleanup
        sys.path.remove(str(schema_path.parent))

    return module


def get_schemas(file: Path) -> Dict[Literal["requirer", "provider"], Type[DataBagSchema]]:
    """Load databag schemas from schema.py file."""
    if not file.exists():
        logger.warning("File does not exist: %s" % file)
        return {}

    try:
        module = load_schema_module(file)
    except ImportError as e:
        logger.error("Failed to load module %s: %s" % (file, e))
        return {}

    out = {}
    for role, name in (("provider", "ProviderSchema"), ("requirer", "RequirerSchema")):
        try:
            out[role] = get_schema_from_module(module, name)
        except NameError:
            logger.warning(
                "Failed to load %s from %s: schema not defined for role: %s." % (name, file, role)
            )
        except TypeError as e:
            logger.error(
                "Found object called %s in %s; expecting a DataBagSchema subclass, not %s."
                % (name, file, e.args[0])
            )
    return out


def _gather_charms_for_version(version_dir: Path) -> Optional[_InterfacesDotYamlSpec]:
    """Attempt to read the `interface.yaml` for this version_dir.

    On failure, return None.
    """
    interface_yaml = version_dir / "interface.yaml"
    if not interface_yaml.exists():
        return None

    charms = None
    try:
        charms = yaml.safe_load(interface_yaml.read_text())
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        logger.error("failed to decode %s: verify that it is valid yaml: %s" % (interface_yaml, e))
    except FileNotFoundError as e:
        logger.error("not found: %s" % e)
    if not charms:
        return None

    providers = charms.get("providers") or []
    requirers = charms.get("requirers") or []
    maintainer = charms.get("maintainer") or ""

    if not isinstance(providers, list) or not isinstance(requirers, list):
        raise TypeError(
            f"{interface_yaml} file has unexpected providers/requirers spec; "
            f"expected two lists of dicts (yaml mappings); "
            f"got {type(providers)}/{type(requirers)}. "
            f"Invalid interface.yaml format."
        )

    provider_configs = []
    requirer_configs = []
    for source, destination in ((providers, provider_configs), (requirers, requirer_configs)):
        for item in source:
            try:
                cfg = _CharmTestConfig(**item)
            except TypeError:
                logger.error(
                    "failure parsing %s to _CharmTestConfig; invalid charm test "
                    "configuration in %s/interface.yaml:providers" % (item, version_dir)
                )
                continue
            destination.append(cfg)

    spec: _InterfacesDotYamlSpec = {
        "providers": provider_configs,
        "requirers": requirer_configs,
        "maintainer": maintainer,
    }
    return spec


def _scrape_module_for_tests(module: types.ModuleType) -> List[Callable[[None], None]]:
    tests = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            tests.append(obj)
    return tests


def _gather_test_cases_for_version(
    version_dir: Path, interface_name: str, version: int, *, tests_dir: str = _DEFAULT_TESTS_DIR
):
    """Collect interface test cases from a directory containing an interface version spec."""

    interface_tests_dir = version_dir / tests_dir

    provider_test_cases = []
    requirer_test_cases = []

    if interface_tests_dir.exists():
        # so we can import without tricks
        sys.path.append(str(interface_tests_dir))

        for role in Role:
            module_name = "test_requirer" if role is Role.requirer else "test_provider"
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                logger.warning("Failed to load module %s: %s" % (module_name, e))
                continue

            tests = _scrape_module_for_tests(module)

            del sys.modules[module_name]

            tgt = provider_test_cases if role is Role.provider else requirer_test_cases
            tgt.extend(tests)

        if not (requirer_test_cases or provider_test_cases):
            logger.error("no valid test case files found in %s" % interface_tests_dir)

        # remove from import search path
        sys.path.pop(-1)

    return provider_test_cases, requirer_test_cases


def gather_test_spec_for_version(
    version_dir: Path, interface_name: str, version: int, *, tests_dir: str = _DEFAULT_TESTS_DIR
) -> InterfaceTestSpec:
    """Collect interface tests from an interface/version subdirectory.

    Given a directory containing an interface specification (conform the template),
    collect and return the interface tests for this version.
    """

    provider_test_cases, requirer_test_cases = _gather_test_cases_for_version(
        version_dir, interface_name, version, tests_dir=tests_dir
    )
    schemas = get_schemas(version_dir / "schema.py")
    charms = _gather_charms_for_version(version_dir)

    return {
        "provider": {
            "tests": provider_test_cases,
            "schema": schemas.get("provider"),
            "charms": charms.get("providers", []) if charms else [],
        },
        "requirer": {
            "tests": requirer_test_cases,
            "schema": schemas.get("requirer"),
            "charms": charms.get("requirers", []) if charms else [],
        },
        "maintainer": charms.get("maintainer") or "" if charms else "",
    }


def _gather_tests_for_interface(
    interface_dir: Path, interface_name: str, *, tests_dir: str = _DEFAULT_TESTS_DIR
) -> Dict[str, InterfaceTestSpec]:
    """Collect interface tests from an interface subdirectory.

    Given a directory containing an interface specification (conform the template),
    collect and return the interface tests for each available version.
    """
    tests = {}
    for version_dir in interface_dir.glob("v*"):
        try:
            version_n = int(version_dir.name[1:])
        except TypeError:
            logger.error(
                "Unable to parse version %s as an integer. Skipping..." % version_dir.name
            )
            continue
        tests[version_dir.name] = gather_test_spec_for_version(
            version_dir, interface_name, version_n, tests_dir=tests_dir
        )
    return tests


def collect_tests(
    path: Path, include: str = "*", *, tests_dir: str = _DEFAULT_TESTS_DIR
) -> Dict[str, Dict[str, InterfaceTestSpec]]:
    """Gather the test cases collected from this path.

    Returns a dict structured as follows:
    - interface name (e.g. "ingress"):
      - version name (e.g. "v2"):
        - role (e.g. "requirer"):
          - tests: [list of interface_test._InterfaceTestCase]
          - schema: <pydantic.BaseModel>
          - charms:
            - name: foo
              url: www.github.com/canonical/foo
    """
    logger.info("collecting tests from %s: %s" % (path, include))
    tests = {}

    for interface_dir in (path / "interfaces").glob(include):
        interface_dir_name = interface_dir.name
        if interface_dir_name.startswith("__"):  # ignore __template__ and python-dirs
            continue  # skip
        logger.info("collecting tests for interface %s" % interface_dir_name)
        interface_name = interface_dir_name.replace("-", "_")
        tests[interface_name] = _gather_tests_for_interface(
            interface_dir, interface_name, tests_dir=tests_dir
        )

    return tests
