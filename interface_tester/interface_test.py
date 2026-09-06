# Copyright 2022 Canonical Ltd.
# See LICENSE file for licensing details.
import dataclasses
import inspect
import logging
import operator
import re
import typing
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, List, Literal, Optional, Union

import pydantic
from ops.testing import CharmType
from pydantic import ValidationError
from scenario import Context, Relation, State
from scenario.context import CharmEvents
from scenario.state import _DEFAULT_JUJU_DATABAG, _Event

from interface_tester.errors import InvalidTestCaseError, SchemaValidationError

RoleLiteral = Literal["requirer", "provider"]

if typing.TYPE_CHECKING:
    InterfaceNameStr = str
    VersionInt = int
    _SchemaConfigLiteral = Literal["default", "skip", "empty"]
    from interface_tester import DataBagSchema

INTF_NAME_AND_VERSION_REGEX = re.compile(r"/interfaces/(\w+)/v(\d+)/")

_RELATION_EVENT_CONSTRUCTORS = {
    "_relation_changed": CharmEvents.relation_changed,
    "_relation_departed": CharmEvents.relation_departed,
    "_relation_broken": CharmEvents.relation_broken,
    "_relation_joined": CharmEvents.relation_joined,
    "_relation_created": CharmEvents.relation_created,
}

logger = logging.getLogger(__name__)

_has_pydantic_v1 = pydantic.version.VERSION.split(".") <= ["2"]
if _has_pydantic_v1:
    logger.warning(
        "You seem to be using pydantic v1. "
        "Please upgrade to v2, as compatibility may be dropped in a future version of "
        "pytest-interface-tester."
    )


def _validate(model: pydantic.BaseModel, obj: dict):
    if _has_pydantic_v1:
        return model.validate(obj)
    return model.model_validate(obj)


class InvalidTestCase(RuntimeError):
    """Raised if a function decorated with interface_test_case is invalid."""


class Role(str, Enum):
    provider = "provider"
    requirer = "requirer"


@dataclasses.dataclass
class _InterfaceTestContext:
    """Data associated with a single interface test case."""

    interface_name: str
    """The name of the interface that this test is about."""
    endpoint: str
    """Endpoint being tested."""
    version: int
    """The version of the interface that this test is about."""
    role: Role
    charm_type: CharmType
    """Charm class being tested"""
    supported_endpoints: dict
    """Supported relation endpoints."""
    meta: Any
    """Charm metadata.yaml"""
    config: Any
    """Charm config.yaml"""
    actions: Any
    """Charm actions.yaml"""
    test_fn: Callable
    """Test function."""
    state_template: Optional[State]
    """Initial state that this test should be run with, according to the charm."""

    """The role (provider|requirer) that this test is about."""
    schema: Optional["DataBagSchema"] = None
    """Databag schema to validate the output relation with."""
    input_state: Optional[State] = None
    """Initial state that this test should be run with, according to the test."""

    juju_version: Optional[str] = None
    """The juju version Scenario will simulate. Defaults to whatever Scenario's default is."""


def check_test_case_validator_signature(fn: Callable):
    """Verify the signature of a test case validator function.

    Will raise InvalidTestCase if:
    - the number of parameters is not exactly 1
    - the parameter is not positional only or positional/keyword

    Will pop a warning if the one argument is annotated with anything other than scenario.State
    (or no annotation).
    """
    sig = inspect.signature(fn)
    if not len(sig.parameters) == 1:
        raise InvalidTestCase(
            "interface test case validator expects exactly one "
            "positional argument of type State."
        )

    params = list(sig.parameters.values())
    par0 = params[0]
    if par0.kind not in (par0.POSITIONAL_OR_KEYWORD, par0.POSITIONAL_ONLY):
        raise InvalidTestCase(
            "interface test case validator expects the first argument to be positional."
        )

    if par0.annotation not in (par0.empty, State):
        logger.warning(
            "interface test case validator will receive a State as first and "
            "only positional argument."
        )


_TESTER_CTX: Optional[_InterfaceTestContext] = None


@contextmanager
def tester_context(ctx: _InterfaceTestContext):
    global _TESTER_CTX
    _TESTER_CTX = ctx

    try:
        yield
    except Exception:
        tester = Tester.__instance__

        if tester:
            tester._detach()

        _TESTER_CTX = None
        raise

    tester = Tester.__instance__

    if not tester:
        raise NoTesterInstanceError(f"Invalid test: {ctx.test_fn} did not instantiate Tester.")

    try:
        tester._finalize()
    finally:
        tester._detach()
        _TESTER_CTX = None

    if Tester.__instance__:
        raise RuntimeError("cleanup failed, tester instance still bound")


class InvalidTesterRunError(RuntimeError):
    """Raised if Tester is being used incorrectly."""

    def __init__(self, test_name: str, msg: str):
        _msg = f"failed running {test_name}: invalid test. {msg}"
        super().__init__(_msg)


class NoTesterInstanceError(RuntimeError):
    """Raised if no Tester is created within a tester_context scope."""


class NoSchemaError(InvalidTesterRunError):
    """Raised when schemas cannot be validated because there is no schema."""


class Tester:
    __instance__ = None

    def __init__(self, state_in: Optional[State] = None, name: Optional[str] = None):
        """Core interface test specification tool.

        This class is essential to defining an interface test to be used in the
        ``charm-relation-interfaces`` repository. In order to define a valid interface
        test you will need to:

        a) Initialize this class in the scope of an interface test to specify the scenario's
        initial state. Then b) call its ``run`` method to execute scenario, and finally
        c) validate the schema.

        Failing to take any of these three steps will result in an invalid test.
        If an error is raised during execution of these steps, or manually from elsewhere in the
        test scope, the test will fail.

        usage:
        >>> def test_foo_relation_joined():
        >>>     t = Tester(state_in=State()) # specify the initial state
        >>>     state_out = t.run('foo-relation-joined') # run Scenario and get the output state
        >>>     t.assert_schema_valid()  # check that the schema is valid

        You can run assertions on ``state_out``, if the interface specification makes
        claims on its contents.

        Alternatively to calling ``assert_schema_valid``, you can:
        1) define your own schema subclassing ``DataBagSchema``
        >>>     from interface_tester.schema_base import DataBagSchema, BaseModel
        >>>     class CustomAppModel(BaseModel):
        >>>         foo: int
        >>>         bar: str
        >>>
        >>>     class SomeCustomSchema(DataBagSchema):
        >>>         app: CustomAppModel
        And then pass it to assert_schema_valid to override the default schema
        (the one defined in ``schema.py``).
        >>>     t.assert_schema_valid(SomeCustomSchema)
        2) check that all local databags are empty (same as ``assert_schema_valid(DataBagSchema)``)
        >>>     t.assert_relation_data_empty()
        3) skip schema validation altogether if you know better
        >>>     t.skip_schema_validation()

        :param state_in: the input state for this scenario test.
            Will default to the empty ``State()``.
        :param name: the name of the test. Will default to the function's
            identifier (``__name__``).
        """
        # todo: pythonify
        if Tester.__instance__:
            raise RuntimeError("Tester is a singleton.")
        Tester.__instance__ = self

        if not self.ctx:
            raise RuntimeError("Tester can only be initialized inside an interface test context.")

        self._state_in = state_in or State()
        self._test_name = name or self.ctx.test_fn.__name__

        self._state_out = None  # will be State when has_run is true
        self._has_run = False
        self._has_checked_schema = False

    @property
    def _test_id(self) -> str:
        """A name for this test, as descriptive and unique as possible."""
        return f"{self.ctx.interface_name}[{self.ctx.version}]/{self.ctx.role}:{self._test_name}"

    @property
    def ctx(self) -> Optional[_InterfaceTestContext]:
        """The test context, defined by the test caller.

        It exposes information about the charm that is using this test.
        You probably won't need to call this from inside the test definition.

        When called from an interface test scope, is guaranteed(^tm) to return
        ``_InterfaceTestContext``.
        """
        return _TESTER_CTX

    def run(self, event: Union[str, _Event]) -> State:
        """Simulate the emission on an event in the initial state you passed to the initializer.

        Calling this method will run scenario and verify that the charm being tested can handle
        the ``event`` without raising exceptions.

        It returns the output state resulting from this execution, should you want to
        write assertions against it.
        """
        if not self.ctx:
            raise InvalidTesterRunError("tester cannot run: no _TESTER_CTX set")

        state_out = self._run(event)
        self._state_out = state_out
        return state_out

    @property
    def _relations(self) -> List[Relation]:
        """The relations that this test is about."""
        return [r for r in self._state_out.relations if r.interface == self.ctx.interface_name]

    def assert_schema_valid(self, schema: Optional["DataBagSchema"] = None):
        """Check that the local databags of the relations being tested satisfy the default schema.

        Default schema is defined in this-interface/vX/schema.py.
        Override the schema being checked against by passing your own DataBagSchema subclass.
        """

        self._has_checked_schema = True
        if not self._has_run:
            raise InvalidTesterRunError(self._test_id, "call Tester.run() first")

        if schema:
            logger.info("running test with custom schema")
            databag_schema = schema
        else:
            logger.info("running test with built-in schema")
            databag_schema = self.ctx.schema
            if not databag_schema:
                raise NoSchemaError(
                    self._test_id,
                    "No schema found. If this is expected, "
                    "call Tester.skip_schema_validation() instead.",
                )

        errors = []
        for relation in self._relations:
            try:
                _validate(
                    databag_schema,
                    {
                        "unit": relation.local_unit_data,
                        "app": relation.local_app_data,
                    },
                )
            except ValidationError as e:
                errors.append(str(e))
        if errors:
            raise SchemaValidationError(errors)

    def _check_has_run(self):
        if not self._has_run:
            raise InvalidTesterRunError(self._test_id, "Call Tester.run() first.")

    def assert_relation_data_empty(self):
        """Assert that all local databags are empty for the relations being tested."""
        self._check_has_run()
        for relation in self._relations:
            if relation.local_app_data:
                raise SchemaValidationError(
                    f"test {self._test_id}: local app databag not empty for {relation}"
                )

            # remove the default unit databag keys or we'll get false positives.
            local_unit_data_keys = set(relation.local_unit_data).difference(
                set(_DEFAULT_JUJU_DATABAG.keys())
            )

            if local_unit_data_keys:
                raise SchemaValidationError(
                    f"test {self._test_id}: local unit databag not empty for {relation}: "
                    f"found {local_unit_data_keys!r} keys set"
                )
        self._has_checked_schema = True

    def skip_schema_validation(self):
        """Skip schema validation for this test run.

        Only use if you really have to.
        """
        self._check_has_run()
        logger.debug("skipping schema validation")
        self._has_checked_schema = True

    def _finalize(self):
        """Verify that .run() has been called, as well as some schema validation method."""
        if not self._has_run:
            raise InvalidTesterRunError(
                self._test_id, "Test function must call Tester.run() before returning."
            )
        if not self._has_checked_schema:
            raise InvalidTesterRunError(
                self._test_id,
                "Test function must call "
                "Tester.skip_schema_validation(), or "
                "Tester.assert_schema_valid(), or "
                "Tester.assert_relation_data_empty() before returning.",
            )
        self._detach()

    def _detach(self):
        # release singleton
        Tester.__instance__ = None

    def _run(self, event: Union[str, _Event]):
        logger.debug("running %s" % event)
        self._has_run = True

        # this is the input state as specified by the interface tests writer. It can
        # contain elements that are required for the relation interface test to work,
        # typically relation data pertaining to the  relation interface being tested.
        input_state = self._state_in

        # state_template is state as specified by the charm being tested, which the charm
        # requires to function properly. Consider it part of the mocking. For example:
        # some required config, a "happy" status, network information, OTHER relations.
        # Typically, should NOT touch the relation that this interface test is about
        #  -> so we overwrite and warn on conflict: state_template is the baseline,
        state = (
            dataclasses.replace(self.ctx.state_template) if self.ctx.state_template else State()
        )

        relations = self._generate_relations_state(
            state, input_state, self.ctx.supported_endpoints, self.ctx.role
        )
        # State is frozen; replace
        modified_state = dataclasses.replace(state, relations=relations)

        # the Relation instance this test is about:
        relation = next(filter(lambda r: r.interface == self.ctx.interface_name, relations))
        evt: _Event = self._cast_event(event, relation)

        logger.info("collected test for %s with %s" % (self.ctx.interface_name, evt.name))
        return self._run_scenario(evt, modified_state)

    def _run_scenario(self, event: Union[str, _Event], state: State):
        logger.debug("running scenario with state=%s, event=%s" % (state, event))

        kwargs = {}
        if self.ctx.juju_version:
            kwargs["juju_version"] = self.ctx.juju_version

        ctx = Context(
            self.ctx.charm_type,
            meta=self.ctx.meta,
            actions=self.ctx.actions,
            config=self.ctx.config,
            **kwargs,
        )
        return ctx.run(event, state)

    def _cast_event(self, raw_event: Union[str, _Event], relation: Relation):
        if not isinstance(raw_event, (_Event, str)):
            raise InvalidTestCaseError(
                f"Bad interface test specification: event {raw_event} should be a relation event "
                f"string or _Event."
            )

        # Juju spells these with dashes, ops with underscores: accept either.
        name = (raw_event if isinstance(raw_event, str) else raw_event.path).replace("-", "_")
        for suffix, constructor in _RELATION_EVENT_CONSTRUCTORS.items():
            if name.endswith(suffix):
                break
        else:
            raise InvalidTestCaseError(
                f"Bad interface test specification: event {raw_event} is not a relation event."
            )

        if isinstance(raw_event, str):
            return constructor(relation)

        # todo: if the user passes a relation event that is NOT about the relation
        #  interface that this test is about, at this point we are injecting the wrong
        #  Relation instance.
        #  e.g. if in interfaces/foo one wants to test that if 'bar-relation-joined' is
        #  fired... then one would have to pass an Event instance already with its
        #  own Relation.

        # next we need to ensure that the event's .relation is our relation, and that the endpoint
        # in the relation and the event path match that of the charm we're testing.
        charm_event = dataclasses.replace(
            raw_event,
            relation=relation,
            path=relation.endpoint + suffix,
        )

        return charm_event

    @staticmethod
    def _get_endpoint(supported_endpoints: dict, role: Role, interface_name: str):
        endpoints_for_interface = supported_endpoints[role]

        if len(endpoints_for_interface) < 1:
            raise ValueError(f"no endpoint found for {role}/{interface_name}.")
        elif len(endpoints_for_interface) > 1:
            raise ValueError(
                f"Multiple endpoints found for {role}/{interface_name}: "
                f"{endpoints_for_interface}: cannot guess which one it is "
                f"we're supposed to be testing"
            )
        return endpoints_for_interface[0]

    def _generate_relations_state(
        self, state_template: State, input_state: State, supported_endpoints, role: Role
    ) -> List[Relation]:
        """Merge the relations from the input state and the state template into one.

        The charm being tested possibly provided a state_template to define some setup mocking data
        The interface tests also have an input_state. Here we merge them into one relation list to
        be passed to the 'final' State the test will run with.
        """
        interface_name = self.ctx.interface_name

        # determine what charm endpoint we're testing.

        endpoint = self.ctx.endpoint or self._get_endpoint(
            supported_endpoints, role, interface_name=interface_name
        )

        for rel in state_template.relations:
            if rel.interface == interface_name:
                logger.warning(
                    "relation with interface name =%s found in state template. "
                    "This will be overwritten by the relation spec provided by the relation "
                    "interface test case." % interface_name
                )

        def filter_relations(rels: List[Relation], op: Callable):
            return [r for r in rels if op(r.interface, interface_name)]

        # the baseline is: all relations provided by the charm in the state_template,
        # whose interface IS NOT the interface we're testing. We assume the test (input_state) is
        # the ultimate owner of the state when it comes to the interface we're testing.
        # We don't allow the charm to mess with it.
        relations = filter_relations(state_template.relations, op=operator.ne)

        if input_state:
            # if the interface test we're running specified some relations in its input_state,
            # we add those whose interface IS the same as the one we're testing.
            # If other relation interfaces were specified (for whatever reason?),
            # they will be ignored.
            relations_from_input_state = filter_relations(input_state.relations, op=operator.eq)

            # relations that come from the state_template presumably have the right endpoint,
            # but those that we get from interface tests cannot.
            relations_with_endpoint = [
                dataclasses.replace(r, endpoint=endpoint) for r in relations_from_input_state
            ]

            relations.extend(relations_with_endpoint)

            if ignored := filter_relations(input_state.relations, op=operator.ne):
                # this is a sign of a bad test.
                logger.warning(
                    "irrelevant relations specified in input_state for %s/%s."
                    "These will be ignored. details: %s" % (interface_name, role, ignored)
                )

        # if we still don't have any relation matching the interface we're testing, we generate
        # one from scratch.
        if not filter_relations(relations, op=operator.eq):
            # if neither the charm nor the interface specified any custom relation spec for
            # the interface we're testing, we will provide one.
            relations.append(
                Relation(
                    interface=interface_name,
                    endpoint=endpoint,
                )
            )
        logger.debug(
            "%s: merged %s and %s --> relations=%s"
            % (self, input_state, state_template, relations)
        )

        return relations
