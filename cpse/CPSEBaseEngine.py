from typing import IO, Callable, Optional, List, Tuple, Union, Dict, Any, Set, Iterable
from abc import abstractmethod
import collections
import operator
import warnings
import traceback
import functools

import unified_planning as up
from unified_planning.engines import (
    Credits,
    PlanGenerationResult,
    PlanGenerationResultStatus,
)
from unified_planning.engines.mixins import OptimalityGuarantee
from unified_planning.model import (
    OperatorKind,
    Parameter,
    FNode,
    timing,
    ProblemKind,
    Fluent,
    Object,
    Presence,
)
from unified_planning.model.scheduling import SchedulingProblem, Activity
from unified_planning.model.metrics import MinimizeMakespan
from unified_planning.plans import Schedule
from unified_planning.model.types import Type, is_compatible_type

from ortools.sat.python import cp_model


# TODO: complete
credits = Credits(
    "CPSE",
    "FBK PSO Unit",
    "",
    "",
    "GPLv3",
    "",
    "",
)

_ARITHMETIC_OPERATOR_MAP = {
    OperatorKind.PLUS: operator.add,
    OperatorKind.MINUS: operator.sub,
    OperatorKind.TIMES: operator.mul,
    OperatorKind.DIV: operator.floordiv,
}

_BOOL_OPERATOR_MAP = {
    OperatorKind.LE: operator.le,
    OperatorKind.LT: operator.lt,
    OperatorKind.EQUALS: operator.eq,
}

activity_type = collections.namedtuple(
    "activity_type", "start end interval is_present up_activity"
)


class CPSEBaseEngine(up.engines.Engine, up.engines.mixins.OneshotPlannerMixin):
    """
    CPSE Base Engine abstract class.

    This class extends the functionality of `up.engines.Engine` and
    `up.engines.mixins.OneshotPlannerMixin`.

    Attributes:
        lower_bound (int): The minimum bound for variables in the model, defaulting to 0.
        upper_bound (int): The maximum bound for variables in the model, defaulting to INT32_MAX.
        model (cp_model.CpModel): The underlying constraint programming model.
        objects (Dict[Object, int]): A dictionary mapping objects to a unique integer identifier.
    """

    def __init__(self, **kwargs):
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)

        self.lower_bound: int = kwargs.get("lower_bound", 0)
        self.upper_bound: int = kwargs.get("upper_bound", cp_model.INT32_MAX)

    def solve_init(self):
        self.model = cp_model.CpModel()
        self.objects: Dict[Object, int] = {}

        # dictionary of activities keyed by name
        self._activities: Dict[str, activity_type] = {}
        # mapping timepoints, parameters and fluents to the corresponding integer variables in the model
        self._model_vars: Dict[
            Union[timing.Timepoint, Parameter, Fluent, Presence], cp_model.IntVar
        ] = {}
        # mapping fluent to its lower bound and upper bound
        self._fluent_bounds: Dict[str, Tuple[int, int]] = {}
        # mapping fluent expression to its initial value expression
        self._fluent_initial_value: Dict[FNode, FNode] = {}
        # mapping types to objects/parameters
        self._type_objects_mapping: Dict[Type, List[Object]] = {}
        self._type_params_mapping: Dict[Type, List[Parameter]] = {}

        # constraints to be applied just before solving the problem
        self._postponed_constraints: List[Callable] = []

        # counters for generating anonymous variables
        self._bool_var_counter: int = -1
        self._int_var_counter: int = -1

        # cache model variables accessed multiple times
        self._variables_cache: Dict[
            str, Union[cp_model.IntVar, cp_model.IntervalVar]
        ] = {}

    @staticmethod
    def get_credits(**kwargs) -> Optional["Credits"]:
        return credits

    @staticmethod
    def satisfies(optimality_guarantee: OptimalityGuarantee) -> bool:
        return optimality_guarantee == OptimalityGuarantee.SATISFICING

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind()
        supported_kind.set_problem_class("SCHEDULING")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_time("DISCRETE_TIME")
        supported_kind.set_time("INTERMEDIATE_CONDITIONS_AND_EFFECTS")
        supported_kind.set_time("EXTERNAL_CONDITIONS_AND_EFFECTS")
        supported_kind.set_time("TIMED_EFFECTS")
        supported_kind.set_time("TIMED_GOALS")
        supported_kind.set_time("DURATION_INEQUALITIES")
        supported_kind.set_expression_duration("INT_TYPE_DURATIONS")
        supported_kind.set_numbers("BOUNDED_TYPES")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_effects_kind("INCREASE_EFFECTS")
        supported_kind.set_effects_kind("DECREASE_EFFECTS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        supported_kind.set_parameters("BOOL_ACTION_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_parameters("UNBOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_quality_metrics("MAKESPAN")
        return supported_kind

    @abstractmethod
    def check_if_supported_problem(self, problem: "up.model.AbstractProblem"):
        """
        Checks if the given problem is a supported instance of `SchedulingProblem`.
        Raises `NotImplementedError` if unsupported.

        Args:
            problem (up.model.AbstractProblem): The problem instance to validate.

        Raises:
            NotImplementedError: If the problem is not supported.
        """

        raise NotImplementedError

    def new_bool_var(self) -> cp_model.IntVar:
        """
        Adds a new anonymous boolean variable to the model.

        Returns:
            cp_model.IntVar: A new integer variable bounded in [0,1].
        """

        self._bool_var_counter += 1
        return self.model.new_bool_var(f"bool{self._bool_var_counter}")

    def new_int_var(self) -> cp_model.IntVar:
        """
        Adds a new anonymous integer variable to the model.

        Returns:
            cp_model.IntVar: A new integer variable.
        """

        self._int_var_counter += 1
        return self.model.new_int_var(
            self.lower_bound, self.upper_bound, f"int{self._int_var_counter}"
        )

    def bool_and_expression(
        self, bool_vars: List[Union[cp_model.IntVar, cp_model.NotBooleanVariable]]
    ) -> Union[cp_model.IntVar, cp_model.NotBooleanVariable]:
        """
        Creates a boolean variable representing the logical AND of the given boolean variables.

        Args:
            bool_vars (List[Union[cp_model.IntVar, cp_model.NotBooleanVariable]]):
                A list of boolean variables to be combined using a logical AND operation.

        Returns:
            Union[cp_model.IntVar, cp_model.NotBooleanVariable]:
                A boolean variable representing the result of the AND operation.
        """

        assert len(bool_vars) >= 1
        if len(bool_vars) == 1:
            return bool_vars[0]
        bool_var = self.new_bool_var()
        self.model.add_bool_and(bool_vars).only_enforce_if(bool_var)
        self.model.add_bool_or([v.negated() for v in bool_vars]).only_enforce_if(
            bool_var.negated()
        )
        return bool_var

    def bool_or_expression(
        self, bool_vars: List[Union[cp_model.IntVar, cp_model.NotBooleanVariable]]
    ) -> Union[cp_model.IntVar, cp_model.NotBooleanVariable]:
        """
        Creates a boolean variable representing the logical OR of the given boolean variables.

        Args:
            bool_vars (List[Union[cp_model.IntVar, cp_model.NotBooleanVariable]]):
                A list of boolean variables to be combined using a logical OR operation.

        Returns:
            Union[cp_model.IntVar, cp_model.NotBooleanVariable]:
                A boolean variable representing the result of the OR operation.
        """

        assert len(bool_vars) >= 1
        if len(bool_vars) == 1:
            return bool_vars[0]
        bool_var = self.new_bool_var()
        self.model.add_bool_or(bool_vars).only_enforce_if(bool_var)
        self.model.add_bool_and([v.negated() for v in bool_vars]).only_enforce_if(
            bool_var.negated()
        )
        return bool_var

    @functools.cache
    def _fnode_contains_fluents(self, fnode: FNode) -> bool:
        """
        Checks if the specified FNode contains any fluent.

        Args:
            fnode (FNode): The FNode to analyze.

        Returns:
            bool: True if a fluent is found within the given `fnode`, otherwise False.
        """

        stack = [fnode]
        while len(stack) > 0:
            fnode = stack.pop()
            if fnode.is_fluent_exp():
                return True

            if len(fnode.args) > 0:
                for arg in fnode.args:
                    stack.append(arg)

        return False

    def _fluent_exp_contains_parameters(self, fluent_exp: FNode) -> bool:
        """
        Checks if the specified fluent expression contains any parameter.

        Args:
            fluent_exp (FNode): The fluent expression to analyze.

        Returns:
            bool: True if the fluent expression contains parameters, otherwise False.
        """

        assert fluent_exp.is_fluent_exp()
        return any(arg.is_parameter_exp() for arg in fluent_exp.args)

    def extract_all_fluent_exp_from_fnode(self, fnode: FNode) -> Set[FNode]:
        """
        Extracts all fluent expressions from the given FNode.

        This method traverses the provided FNode recursively and identifies all
        fluent expressions contained within it.

        Args:
            fnode (FNode): The FNode to be traversed.

        Returns:
            Set[FNode]: A set of fluent expressions (`FNode`) found within the given `fnode`.
        """

        all_fluent_exps = set()
        stack = [fnode]
        while len(stack) > 0:
            fnode = stack.pop()
            if fnode.is_fluent_exp():
                all_fluent_exps.add(fnode)
            elif len(fnode.args) > 0:
                for arg in fnode.args:
                    stack.append(arg)

        return all_fluent_exps

    def extract_all_parametric_fluent_exp_from_fnode(self, fnode: FNode) -> Set[FNode]:
        """
        Extracts all fluent expressions with parameters from the given FNode.

        This method traverses the provided FNode recursively and identifies all
        fluent expressions with parameters contained within it.

        Args:
            fnode (FNode): The FNode to be traversed.

        Returns:
            Set[FNode]: A set of fluent expressions (`FNode`) found within the given `fnode`.
        """

        return set(
            filter(
                lambda fluent_exp: self._fluent_exp_contains_parameters(fluent_exp),
                self.extract_all_fluent_exp_from_fnode(fnode),
            )
        )

    def extract_all_params_from_fluent_exps(
        self, fluent_exps: Iterable[FNode]
    ) -> Set[Parameter]:
        """
        Extracts all parameters from the given fluent expressions.

        This method iterates through the provided fluent expressions and collects
        all unique parameters used within them.

        Args:
            fluent_exps (Iterable[FNode]): An iterable of fluent expressions from which parameters will be extracted.

        Returns:
            Set[Parameter]: A set of unique parameters found within the given fluent expressions.
        """

        all_parameters = set()
        for fluent_exp in fluent_exps:
            for arg in fluent_exp.args:
                if arg.is_parameter_exp():
                    all_parameters.add(arg.parameter())
        return all_parameters

    def _convert_timing_to_linear_expr(self, t: timing.Timing) -> cp_model.LinearExprT:
        """
        Converts a `timing.Timing` variable to a `cp_model.LinearExprT`.

        Args:
            t (timing.Timing): The timing variable.

        Returns:
            cp_model.LinearExprT: The corresponding linear expression in the model.
        """

        assert t.timepoint in self._model_vars
        assert isinstance(t.delay, int)
        return cp_model.LinearExpr.sum([self._model_vars[t.timepoint], t.delay])

    def _get_timeinterval_lower_upper_bounds(
        self, time_interval: timing.TimeInterval
    ) -> Tuple[cp_model.LinearExprT, cp_model.LinearExprT]:
        """
        Returns the linear expressions representing the lower and upper
        bounds of the specified time interval in the model

        Args:
            time_interval (timing.TimeInterval): The time interval for which the bounds
            are to be determined.

        Returns:
            Tuple[cp_model.LinearExprT, cp_model.LinearExprT]: A tuple containing:
                - The first element is the lower bound of the time interval.
                - The second element is the upper bound of the time interval.
        """

        lower_delay = 0
        if time_interval.lower.delay != 0:
            lower_delay += time_interval.lower.delay
        if time_interval.is_left_open():
            lower_delay += 1
        lower = cp_model.LinearExpr.sum(
            [self._model_vars[time_interval.lower.timepoint], lower_delay]
        )

        upper_delay = 0
        if time_interval.upper.delay != 0:
            upper_delay += time_interval.upper.delay
        if time_interval.is_right_open():
            upper_delay -= 1
        upper = cp_model.LinearExpr.sum(
            [self._model_vars[time_interval.upper.timepoint], upper_delay]
        )

        return lower, upper

    def _get_lower_upper_bounds(self, fluent: Fluent) -> Tuple[int, int]:
        """
        Determines the lower and upper bounds for a given fluent based on its type.

        Args:
            fluent (Fluent): The fluent for which bounds are to be determined. Must have
                a type that is either integer or boolean.

        Returns:
            Tuple[int, int]: A tuple containing the lower and upper bounds for the fluent.

        Raises:
            NotImplementedError: If the fluent's type is not integer or boolean.
        """

        # TODO: use self.lower_bound or INT32_MIN?

        if fluent.type.is_int_type():
            lower_bound = fluent.type.lower_bound
            if fluent.type.lower_bound is None:
                lower_bound = self.lower_bound

            upper_bound = fluent.type.upper_bound
            if fluent.type.upper_bound is None:
                upper_bound = self.upper_bound

        elif fluent.type.is_bool_type():
            lower_bound = 0
            upper_bound = 1

        else:
            raise NotImplementedError("Only integer and boolean fluents are supported.")

        return lower_bound, upper_bound

    def process_fluents(self, problem: SchedulingProblem):
        """
        Maps each fluent to its lower bound, upper bound and initial value.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the fluents.
        """

        self._fluent_bounds = {}
        for fluent in problem.fluents:
            lower_bound, upper_bound = self._get_lower_upper_bounds(fluent)
            self._fluent_bounds[fluent.name] = (lower_bound, upper_bound)

        self._fluent_initial_value = {}
        for fluent_exp, initial_value in problem.initial_values.items():
            assert fluent_exp.is_fluent_exp()
            if initial_value.is_int_constant():
                lower_bound, upper_bound = self._fluent_bounds[fluent_exp.fluent().name]
                assert lower_bound <= initial_value.int_constant_value() <= upper_bound
            self._fluent_initial_value[fluent_exp] = initial_value

    def process_objects(self, problem: SchedulingProblem):
        """
        Maps objects in the scheduling problem to their respective types and assigns
        each a unique integer value.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the objects.
        """

        for type in problem.user_types:
            self._type_objects_mapping[type] = []
        for obj in problem.all_objects:
            if not obj.type.is_user_type():
                continue
            self.objects[obj] = len(self._type_objects_mapping[obj.type])
            self._type_objects_mapping[obj.type].append(obj)

    def add_parameters(self, problem: SchedulingProblem):
        """
        Adds the parameters to the model as boolean or integer variables.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                parameters to be added to the model.
        """

        for type in problem.user_types:
            self._type_params_mapping[type] = []

        activity_parameters = []
        for activity in problem.activities:
            activity_parameters += activity.parameters

        for param in problem.base_variables + activity_parameters:
            if param.type.is_bool_type():
                var = self.model.new_bool_var(param.name)
            elif param.type.is_int_type():
                lb = (
                    self.lower_bound
                    if param.type.lower_bound is None
                    else param.type.lower_bound
                )
                ub = (
                    self.upper_bound
                    if param.type.upper_bound is None
                    else param.type.upper_bound
                )
                var = self.model.new_int_var(lb, ub, param.name)
            elif param.type.is_user_type():
                lb = 0
                ub = len(list(problem.objects(param.type))) - 1
                var = self.model.new_int_var(lb, ub, param.name)
                self._type_params_mapping[param.type].append(param)
            else:
                raise NotImplementedError(f"Parameter type {param.type} not supported.")

            assert param not in self._model_vars
            self._model_vars[param] = var

    def add_presence_expressions(self, problem: SchedulingProblem):
        for activity in problem.activities:
            if activity.optional:
                present_exp = activity.present.presence()
                var = self.model.new_bool_var(f"present({present_exp.container})")

                assert present_exp not in self._model_vars
                self._model_vars[present_exp] = var

    def _get_compatible_objects(self, type: Type) -> List[Object]:
        """
        Retrieves a list of objects that are compatible with the specified type.

        Args:
            type (Type): The type against which compatibility of objects is checked.

        Returns:
            List[Object]: A list of `Object` instances that are compatible with the given `type`.
        """

        objects = []
        for other_type in self._type_objects_mapping:
            if is_compatible_type(type, other_type):
                objects += self._type_objects_mapping[other_type]
        return objects

    def _get_compatible_parameters(self, type: Type) -> List[Parameter]:
        """
        Retrieves a list of parameters that are compatible with the specified type.

        Args:
            type (Type): The type against which compatibility of parameters is checked.

        Returns:
            List[Parameter]: A list of `Parameter` instances that are compatible with the given `type`.
        """

        parameters = []
        for other_type in self._type_params_mapping:
            if is_compatible_type(type, other_type):
                parameters += self._type_params_mapping[other_type]
        return parameters

    def _add_activity_timepoints(self, activity: Activity) -> Tuple[
        Union[int, cp_model.IntVar],
        Union[int, cp_model.IntVar],
        Union[int, cp_model.IntVar],
    ]:
        """
        Adds variables for the start, end, and duration of an activity, and enforces
        constraints on them.

        Args:
            activity (Activity): The activity for which variables are being defined.

        Returns:
            Tuple[Union[int, cp_model.IntVar], Union[int, cp_model.IntVar], Union[int, cp_model.IntVar]]:
                A tuple containing the start time, end time, and duration for the activity,
                each represented as either an integer (for fixed values) or a model variable.
        """

        assert (
            not activity.duration.is_left_open()
            and not activity.duration.is_right_open()
        )
        start_var = self.model.new_int_var(
            self.lower_bound, self.upper_bound, "start_" + activity.name
        )
        end_var = self.model.new_int_var(
            self.lower_bound, self.upper_bound, "end_" + activity.name
        )
        lower = activity.duration.lower
        upper = activity.duration.upper
        if lower.is_int_constant() and upper.is_int_constant():
            lower = lower.int_constant_value()
            upper = upper.int_constant_value()
            if lower == upper:
                # FixedDuration
                duration_var = upper
            else:
                # ClosedDurationInterval
                duration_var = self.model.new_int_var(
                    self.lower_bound, self.upper_bound, "duration_" + activity.name
                )
                self.model.add_linear_constraint(duration_var, lower, upper)
        else:
            duration_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "duration_" + activity.name
            )
            if lower == upper:
                # FixedDuration
                duration_exp = self.fnode_to_value_or_variable(upper)
                self.model.add(duration_var == duration_exp)
            else:
                # ClosedDurationInterval
                lower_exp = self.fnode_to_value_or_variable(lower)
                upper_exp = self.fnode_to_value_or_variable(upper)
                self.model.add(lower_exp <= duration_var)
                self.model.add(duration_var <= upper_exp)

        return start_var, duration_var, end_var

    def add_activity(self, activity: Activity):
        """
        Adds an activity to the model. Each activity is modeled using an interval.

        Args:
            activity (Activity): The activity to be added to the model.
        """

        start_var, duration_var, end_var = self._add_activity_timepoints(activity)
        if activity.optional:
            is_present_var = self._model_vars[activity.present.presence()]
            interval_var = self.model.new_optional_interval_var(
                start_var, duration_var, end_var, is_present_var, activity.name
            )
        else:
            is_present_var = None
            interval_var = self.model.new_interval_var(
                start_var, duration_var, end_var, activity.name
            )
        self._activities[activity.name] = activity_type(
            start=start_var,
            end=end_var,
            interval=interval_var,
            is_present=is_present_var,
            up_activity=activity,
        )
        self._model_vars[activity.start] = start_var
        self._model_vars[activity.end] = end_var
        self._variables_cache[f"interval [{activity.start}, {activity.end}]"] = (
            interval_var
        )

    def fnode_to_value_or_variable(
        self, fnode: FNode
    ) -> Union[cp_model.IntVar, cp_model.LinearExprT, bool, int, Any]:
        """
        Converts an FNode to its corresponding value or variable representation
        in the model.

        This method processes the given FNode and returns the corresponding
        value or variable. The FNode may represent a constant value, a boolean
        expression, an integer expression, a parameter expression, a timing
        expression, or a fluent expression.

        Args:
            fnode (FNode): The FNode representing an expression.

        Returns:
            Union[cp_model.IntVar, cp_model.LinearExprT, bool, int, Any]:
                The corresponding value or variable for the FNode. This can be
                an `IntVar`, integer, boolean, a linear expression or the result
                of applying an operator.
        """

        stack = [(fnode, False)]
        results = []

        while len(stack) > 0:
            fnode, processed = stack.pop()

            if fnode.is_parameter_exp():
                results.append(self._model_vars[fnode.parameter()])

            elif fnode.is_presence_exp():
                results.append(self._model_vars[fnode.presence()])

            elif fnode.is_object_exp():
                obj = fnode.object()
                results.append(self.objects[obj])

            elif fnode.is_timing_exp():
                results.append(self._convert_timing_to_linear_expr(fnode.timing()))

            elif fnode.is_fluent_exp():
                if fnode not in self._model_vars:
                    raise NotImplementedError(
                        f"Node type {fnode.node_type} not supported."
                    )
                results.append(self._model_vars[fnode])

            elif fnode.node_type in [
                OperatorKind.BOOL_CONSTANT,
                OperatorKind.INT_CONSTANT,
            ]:
                results.append(fnode.constant_value())

            elif fnode.node_type in _ARITHMETIC_OPERATOR_MAP:
                if not processed:
                    stack.append((fnode, True))
                    for arg in fnode.args:
                        stack.append((arg, False))
                else:
                    op = _ARITHMETIC_OPERATOR_MAP[fnode.node_type]
                    args = [results.pop() for arg in fnode.args]
                    results.append(op(*args))

            else:
                raise NotImplementedError(f"Node type {fnode.node_type} not supported.")

        assert len(results) == 1
        return results[0]

    def add_constraint(
        self, fnode: FNode, cache_enabled: bool = True
    ) -> Union[cp_model.IntVar, cp_model.NotBooleanVariable]:
        """
        Adds the constraint represented by the given FNode to the model.

        The constraint expression is processed recursively from the leaves of
        the expression tree to the root node. For each boolean expression, a
        boolean variable is created to represent the expression itself, which
        is then used in the parent node.

        Args:
            fnode (FNode): The FNode representing the constraint to be added.
                The constraint must be a boolean expression.
            cache_enabled (bool): Whether to use cached values for efficiency
                (default is True).

        Returns:
            Union[cp_model.IntVar, cp_model.NotBooleanVariable]:
                A boolean variable (or its negation) representing the constraint.
        """

        # force disabling cache when fnode includes fluents
        if self._fnode_contains_fluents(fnode):
            cache_enabled = False

        stack = [(fnode, False)]
        results = []

        while len(stack) > 0:
            fnode, processed = stack.pop()

            # check if fnode cached
            if cache_enabled and not processed and repr(fnode) in self._variables_cache:
                results.append(self._variables_cache[repr(fnode)])

            elif fnode.node_type in [
                OperatorKind.PARAM_EXP,
                OperatorKind.IS_PRESENT_EXP,
                OperatorKind.OBJECT_EXP,
                OperatorKind.TIMING_EXP,
                OperatorKind.FLUENT_EXP,
                OperatorKind.BOOL_CONSTANT,
                OperatorKind.INT_CONSTANT,
                OperatorKind.PLUS,
                OperatorKind.MINUS,
                OperatorKind.TIMES,
                OperatorKind.DIV,
            ]:
                results.append(self.fnode_to_value_or_variable(fnode))

            elif fnode.node_type == OperatorKind.NOT:
                if not processed:
                    stack.append((fnode, True))
                    stack.append((fnode.args[0], False))
                else:
                    results.append(results.pop().negated())

            elif fnode.node_type in [
                OperatorKind.AND,
                OperatorKind.OR,
                OperatorKind.IMPLIES,
                OperatorKind.IFF,
            ]:
                if not processed:
                    stack.append((fnode, True))
                    for arg in fnode.args:
                        stack.append((arg, False))
                    continue

                args = [results.pop() for arg in fnode.args]
                bool_var = self.new_bool_var()
                if fnode.node_type == OperatorKind.AND:
                    self.model.add_bool_and(args).only_enforce_if(bool_var)
                    self.model.add_bool_or([b.negated() for b in args]).only_enforce_if(
                        bool_var.negated()
                    )
                elif fnode.node_type == OperatorKind.OR:
                    self.model.add_bool_or(args).only_enforce_if(bool_var)
                    self.model.add_bool_and(
                        [b.negated() for b in args]
                    ).only_enforce_if(bool_var.negated())
                elif fnode.node_type == OperatorKind.IMPLIES:
                    assert len(args) == 2
                    self.model.add_implication(args[0], args[1]).only_enforce_if(
                        bool_var
                    )
                    self.model.add_bool_and(args[0], args[1].negated()).only_enforce_if(
                        bool_var.negated()
                    )
                elif fnode.node_type == OperatorKind.IFF:
                    assert len(args) == 2
                    self.model.add(args[0] == args[1]).only_enforce_if(bool_var)
                    self.model.add(args[0] != args[1]).only_enforce_if(
                        bool_var.negated()
                    )

                results.append(bool_var)
                if cache_enabled:
                    self._variables_cache[repr(fnode)] = bool_var

            elif fnode.node_type in _BOOL_OPERATOR_MAP:
                if not processed:
                    assert len(fnode.args) == 2
                    stack.append((fnode, True))
                    for arg in fnode.args:
                        stack.append((arg, False))
                    continue

                args = [results.pop() for arg in fnode.args]
                op = _BOOL_OPERATOR_MAP[fnode.node_type]
                bool_var = self.new_bool_var()
                self.model.add(op(args[0], args[1])).only_enforce_if(bool_var)
                if fnode.node_type == OperatorKind.EQUALS:
                    self.model.add(args[0] != args[1]).only_enforce_if(
                        bool_var.negated()
                    )
                elif fnode.node_type == OperatorKind.LE:
                    self.model.add(args[0] > args[1]).only_enforce_if(
                        bool_var.negated()
                    )
                elif fnode.node_type == OperatorKind.LT:
                    self.model.add(args[0] >= args[1]).only_enforce_if(
                        bool_var.negated()
                    )

                results.append(bool_var)
                if cache_enabled:
                    self._variables_cache[repr(fnode)] = bool_var

            else:
                raise NotImplementedError(f"Node type {fnode.node_type} not supported.")

        assert len(results) == 1

        # FIXME
        if isinstance(results[0], bool):
            bool_var = self.new_bool_var()
            self.model.add_bool_and(results[0]).only_enforce_if(bool_var)
            return bool_var
        assert isinstance(results[0], cp_model.IntVar) or isinstance(
            results[0], cp_model.NotBooleanVariable
        )
        return results[0]

    @abstractmethod
    def add_constraints(self, problem: SchedulingProblem):
        """
        Adds all constraints from the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                constraints to be added to the model.
        """

        raise NotImplementedError

    @abstractmethod
    def add_effects(self, problem: SchedulingProblem):
        """
        Adds the effects of the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                effects to be applied to the model.
        """

        raise NotImplementedError

    @abstractmethod
    def add_conditions(self, problem: SchedulingProblem):
        """
        Adds the conditions of the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                conditions to be added to the model.
        """

        raise NotImplementedError

    def add_quality_metrics(
        self, problem: SchedulingProblem, makespan_var: cp_model.IntVar
    ):
        """
        Adds quality metrics to the model based on the given scheduling problem.

        Currently, only minimizing the makespan is supported as a quality metric.

        Args:
            problem (SchedulingProblem): The scheduling problem containing
                the quality metrics.
            makespan_var (cp_model.IntVar): The variable representing the
                makespan of the schedule, used to define the quality metrics.
        """

        for metric in problem.quality_metrics:
            if isinstance(metric, MinimizeMakespan):
                self.model.minimize(makespan_var)
            else:
                raise NotImplementedError(f"Quality metric {metric} not supported.")

    def _solve(
        self,
        problem: "up.model.AbstractProblem",
        heuristic: Optional[Callable[["up.model.state.State"], Optional[float]]] = None,
        timeout: Optional[float] = None,
        output_stream: Optional[IO[str]] = None,
    ) -> "up.engines.results.PlanGenerationResult":

        self.solve_init()

        try:
            self.check_if_supported_problem(problem)

            if heuristic is not None:
                warnings.warn("CPSE does not support custom heuristics", UserWarning)

            self.model.name = problem.name

            self.process_objects(problem)
            self.process_fluents(problem)

            # add problem-specific and activity-related parameters to the model
            self.add_parameters(problem)
            self.add_presence_expressions(problem)

            # add the activities to the model
            for activity in problem.activities:
                self.add_activity(activity)

            # define the makespan variable
            makespan_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "makespan"
            )
            self.model.add_max_equality(
                makespan_var, [a.end for a in self._activities.values()]
            )

            # add global start and end to the model variables
            global_start = timing.GlobalStartTiming(delay=0).timepoint
            global_end = timing.GlobalEndTiming().timepoint
            self._model_vars[global_start] = 0
            self._model_vars[global_end] = makespan_var

            # add problem-specific and activity-related effects to the model
            self.add_effects(problem)
            # add problem-specific and activity-related constraints to the model
            self.add_constraints(problem)
            # add problem-specific and activity-related conditions to the model
            self.add_conditions(problem)

            # add quality metrics to the model
            self.add_quality_metrics(problem, makespan_var)

            for postponed_constraint in self._postponed_constraints:
                postponed_constraint()

            # solve the modeled problem
            solver = cp_model.CpSolver()
            if timeout is not None:
                solver.parameters.max_time_in_seconds = timeout
            if output_stream is not None:
                solver.parameters.log_search_progress = True
                solver.log_callback = lambda s: output_stream.write(s + "\n")
                solver.parameters.log_to_stdout = False

            status = solver.solve(self.model)

            # define metrics to be returned with the result
            metrics = {
                "wall_time": str(solver.wall_time),
                "user_time": str(solver.user_time),
                "objective_value": str(solver.objective_value),
                "best_objective_bound": str(solver.best_objective_bound),
                "branches": str(solver.num_branches),
                "conflicts": str(solver.num_conflicts),
                "num_booleans": str(solver.num_booleans),
            }

            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                # map a decision variable to its solution value
                assignment = {}
                for up_var, cp_var in self._model_vars.items():
                    # map a boolean parameter to its boolean value rather than the
                    # integer value returned by the solver
                    if isinstance(up_var, Parameter):
                        if up_var.type.is_bool_type():
                            assert solver.value(cp_var) in [0, 1]
                            assignment[up_var] = solver.value(cp_var) == 1
                        elif up_var.type.is_user_type():
                            for obj in self._type_objects_mapping[up_var.type]:
                                if self.objects[obj] == solver.value(cp_var):
                                    assignment[up_var] = obj
                                    break
                        else:
                            assignment[up_var] = solver.value(cp_var)

                    elif isinstance(up_var, Presence):
                        assert solver.value(cp_var) in [0, 1]
                        assignment[up_var] = solver.value(cp_var) == 1

                    elif isinstance(up_var, timing.Timepoint) and up_var not in [
                        global_start,
                        global_end,
                    ]:
                        assignment[up_var] = solver.value(cp_var)

                # filter optional activities not present in the plan
                activities = list(
                    filter(
                        lambda act: not act.optional
                        or assignment[act.present.presence()],
                        problem.activities,
                    )
                )
                plan = Schedule(activities, assignment, problem.environment)

                if status == cp_model.OPTIMAL and len(problem.quality_metrics) > 0:
                    result_status = PlanGenerationResultStatus.SOLVED_OPTIMALLY
                else:
                    result_status = PlanGenerationResultStatus.SOLVED_SATISFICING

                return PlanGenerationResult(
                    result_status,
                    plan,
                    engine_name=self.name,
                    log_messages=None,
                    metrics=metrics,
                )
            else:
                if status == cp_model.INFEASIBLE:
                    result_status = PlanGenerationResultStatus.UNSOLVABLE_PROVEN
                elif status == cp_model.UNKNOWN:
                    result_status = PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
                    if timeout is not None and solver.wall_time > timeout:
                        result_status = PlanGenerationResultStatus.TIMEOUT
                elif status == cp_model.MODEL_INVALID:
                    raise Exception("The CP-SAT model is incorrectly specified.")

                return PlanGenerationResult(
                    result_status,
                    plan=None,
                    engine_name=self.name,
                    log_messages=None,
                    metrics=metrics,
                )

        except NotImplementedError as e:
            return PlanGenerationResult(
                PlanGenerationResultStatus.UNSUPPORTED_PROBLEM,
                plan=None,
                engine_name=self.name,
                log_messages=e,
            )

        except:
            return PlanGenerationResult(
                PlanGenerationResultStatus.INTERNAL_ERROR,
                plan=None,
                engine_name=self.name,
                log_messages=traceback.format_exc(),
            )
