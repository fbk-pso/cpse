#!/usr/bin/env python3
"""Unified Planning Integration for OR-Tools CP-SAT Model"""
from typing import IO, Callable, Optional, List, Tuple, Union, Dict, Set, Iterable, Any
from abc import abstractmethod
import collections
import operator
import warnings
import traceback
import functools
import itertools

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
    Effect,
    Fluent,
    Object,
)
from unified_planning.model.scheduling import SchedulingProblem, Activity
from unified_planning.model.metrics import MinimizeMakespan
from unified_planning.plans import Schedule
from unified_planning.shortcuts import Plus, Minus
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
    "activity_type", "start end interval up_activity"
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
    """

    def __init__(self, **kwargs):
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)

        self.lower_bound: int = kwargs.get("lower_bound", 0)
        self.upper_bound: int = kwargs.get("upper_bound", cp_model.INT32_MAX)
        self.model = cp_model.CpModel()

        # dictionary of activities keyed by name
        self._activities: Dict[str, activity_type] = {}
        # mapping timepoints, parameters and fluents to the corresponding integer variables in the model
        self._model_vars: Dict[
            Union[timing.Timepoint, Parameter, Fluent], cp_model.IntVar
        ] = {}
        # mapping fluent to its lower bound and upper bound
        self._fluent_bounds: Dict[str, Tuple[int, int]] = {}
        # mapping fluent expression to its initial value expression
        self._fluent_initial_value: Dict[FNode, FNode] = {}
        # mapping objects to their types and assigns a unique integer value
        self.objects: Dict[Object, int] = {}
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
        lower = activity.duration.lower
        upper = activity.duration.upper
        if lower.is_int_constant() and upper.is_int_constant():
            lower = lower.int_constant_value()
            upper = upper.int_constant_value()
            if lower == upper:
                # FixedDuration
                start_var = self.model.new_int_var(
                    self.lower_bound, self.upper_bound, "start_" + activity.name
                )
                end_var = self.model.new_int_var(
                    self.lower_bound, self.upper_bound, "end_" + activity.name
                )
                duration_var = upper
            else:
                # ClosedDurationInterval
                start_var = lower
                end_var = upper
                duration_var = upper - lower
        else:
            start_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "start_" + activity.name
            )
            end_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "end_" + activity.name
            )
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
                self.model.add(start_var == lower_exp)
                upper_exp = self.fnode_to_value_or_variable(upper)
                self.model.add(end_var == upper_exp)

        return start_var, duration_var, end_var

    def add_activity(self, activity: Activity):
        """
        Adds an activity to the model. Each activity is modeled using an interval.

        Args:
            activity (Activity): The activity to be added to the model.
        """

        start_var, duration_var, end_var = self._add_activity_timepoints(activity)
        interval_var = self.model.new_interval_var(
            start_var, duration_var, end_var, activity.name
        )
        self._activities[activity.name] = activity_type(
            start=start_var, end=end_var, interval=interval_var, up_activity=activity
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
    ) -> Union[cp_model.IntVar, cp_model._NotBooleanVariable]:
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
            Union[cp_model.IntVar, cp_model._NotBooleanVariable]:
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
        assert isinstance(results[0], cp_model.IntVar) or isinstance(
            results[0], cp_model._NotBooleanVariable
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

        try:
            self.check_if_supported_problem(problem)

            if heuristic is not None:
                warnings.warn("CPSE does not support custom heuristics", UserWarning)

            self.model.name = problem.name

            self.process_objects(problem)
            self.process_fluents(problem)

            # add problem-specific and activity-related parameters to the model
            self.add_parameters(problem)

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
            self._model_vars[timing.GlobalStartTiming(delay=0).timepoint] = 0
            self._model_vars[timing.GlobalEndTiming().timepoint] = makespan_var

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
                    assignment[up_var] = solver.value(cp_var)
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

                plan = Schedule(problem.activities, assignment, problem.environment)

                if status == cp_model.OPTIMAL:
                    result_status = PlanGenerationResultStatus.SOLVED_OPTIMALLY
                else:
                    result_status = PlanGenerationResultStatus.SOLVED_SATISFICING
                    if timeout is not None and solver.wall_time > timeout:
                        result_status = PlanGenerationResultStatus.TIMEOUT

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


class CPSE(CPSEBaseEngine):
    """
    Implementation of the CPSE Engine.

    This class extends the `CPSEBaseEngine` and provides concrete implementations
    for the abstract methods defined in the base class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "CPSE"

    @staticmethod
    def supports(problem_kind: ProblemKind) -> bool:
        return problem_kind <= CPSE.supported_kind()

    def check_if_supported_problem(self, problem: "up.model.AbstractProblem"):
        """
        Checks if the given problem is a supported instance of `SchedulingProblem`.
        Raises `NotImplementedError` if unsupported.

        Args:
            problem (up.model.AbstractProblem): The problem instance to validate.

        Raises:
            NotImplementedError: If the problem is not supported.
        """

        # TODO: add more checks

        if not isinstance(problem, SchedulingProblem):
            raise NotImplementedError(f"Problem of type {type(problem)} not supported.")
        problem: SchedulingProblem

        if not problem.discrete_time:
            raise NotImplementedError("Continuous time not supported.")

        for timing, eff, activity in problem.all_effects():
            if self._fluent_exp_contains_parameters(eff.fluent):
                raise NotImplementedError("Fluents with parameters are not supported.")

    def add_constraints(self, problem: SchedulingProblem):
        """
        Adds all constraints from the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                constraints to be added to the model.
        """

        # TODO: avoid bool_var for the root node

        for fnode in problem.base_constraints:
            bool_var = self.add_constraint(fnode)
            self.model.add_bool_and([bool_var])

        for act in problem.activities:
            for fnode in act.constraints:
                bool_var = self.add_constraint(fnode)
                self.model.add_bool_and([bool_var])

    def add_effects(self, problem: SchedulingProblem):
        """
        Adds the effects of the given scheduling problem to the model.

        Adds a reservoir constraint for each fluent in the scheduling problem.
        Each fluent's value is constrained to remain within the range [0, C],
        where C is the maximum capacity of the fluent.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                effects to be applied to the model.
        """

        # map each fluent to its effects, adjusting values based on increase/decrease effect types
        fluent_effects: Dict[
            FNode, List[Tuple["timing.Timing", int, cp_model.IntVar]]
        ] = {}
        for timing, eff, activity in problem.all_effects():
            eff: Effect
            fluent_exp = eff.fluent

            if eff.value.is_int_constant():
                value = eff.value.int_constant_value()
            elif eff.value.is_bool_constant():
                value = 1 if eff.value.bool_constant_value() else 0
            else:
                raise NotImplementedError(
                    "Effect values must be constants of type boolean or integer."
                )
            if value == 0:  # if value is 0, the effect is ignored
                continue

            if fluent_exp not in fluent_effects:
                fluent_effects[fluent_exp] = []

            if eff.is_increase():
                pass
            elif eff.is_decrease():
                value = -value
            else:
                raise NotImplementedError(f"Effect kind {eff.kind} not supported.")

            if eff.is_conditional():
                bool_var = self.add_constraint(eff.condition)
                fluent_effects[fluent_exp].append((timing, value, bool_var))
            else:
                fluent_effects[fluent_exp].append((timing, value, True))

        for fluent_exp in fluent_effects:
            lb, ub = self._fluent_bounds[fluent_exp.fluent().name]
            if fluent_exp not in self._fluent_initial_value:
                raise NotImplementedError(
                    f"Fluent '{fluent_exp}' must be initialized with a constant value of type integer or boolean."
                )
            init_value = self._fluent_initial_value[fluent_exp]
            if init_value.is_int_constant():
                init_value = init_value.int_constant_value()
            elif init_value.is_bool_constant():
                if init_value.bool_constant_value() == False:
                    init_value = 0
                else:
                    init_value = 1
            else:
                raise NotImplementedError(
                    "Only integer and boolean constants are supported as initial values for fluents."
                )

            times = [0]
            values = [init_value]
            actives = [True]
            for timing, value, active in fluent_effects[fluent_exp]:
                times.append(self._convert_timing_to_linear_expr(timing))
                values.append(value)
                actives.append(active)

            if lb > 0:
                # TODO: add support for lb > 0
                raise NotImplementedError(
                    "Fluent lower bound cannot be greater than 0."
                )

            self.model.add_reservoir_constraint_with_active(
                times, values, actives, lb, ub
            )

    def add_condition(
        self, time_interval: timing.TimeInterval, fnode: FNode, name: str
    ):
        """
        Adds a condition to the model, enforcing that it is satisfied within a specified time interval.

        This method creates a constraint based on the given condition and uses the `add_cumulative`
        method to ensure it holds throughout the specified time interval.

        Args:
            time_interval (timing.TimeInterval): The time interval during which the condition
                must be satisfied.
            fnode (FNode): The FNode representing the condition to be added as a constraint.
            name (str): The name of the condition for identification within the model.
        """

        bool_var = self.add_constraint(fnode)

        start_delay = 0
        if time_interval.lower.delay != 0:
            start_delay += time_interval.lower.delay
        if time_interval.is_left_open():
            start_delay += 1
        start = self._model_vars[time_interval.lower.timepoint] + start_delay

        # add 1 to end because `add_cumulative` enforces the constraint for t in [start, end),
        # but we want the constraint to be enforced also at the end
        end_delay = 1
        if time_interval.upper.delay != 0:
            end_delay += time_interval.upper.delay
        if time_interval.is_right_open():
            end_delay -= 1
        end = self._model_vars[time_interval.upper.timepoint] + end_delay

        interval_key = f"interval [{time_interval.lower.timepoint} + {start_delay}, {time_interval.upper.timepoint} + {end_delay}]"
        if interval_key not in self._variables_cache:
            duration = self.model.new_int_var(
                self.lower_bound,
                self.upper_bound,
                f"{name}_duration",
            )
            interval_var = self.model.new_interval_var(start, duration, end, name)
            self._variables_cache[interval_key] = interval_var
        interval_var = self._variables_cache[interval_key]
        self.model.add_cumulative([interval_var], [bool_var.negated()], 0)

    def add_conditions(self, problem: SchedulingProblem):
        """
        Adds the conditions of the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                conditions to be added to the model.
        """

        for i, (time_interval, fnode) in enumerate(problem.base_conditions):
            name = f"base_condition{i}"
            self.add_condition(time_interval, fnode, name)

        for act in problem.activities:
            i = 0
            for time_interval in act.conditions:
                for fnode in act.conditions[time_interval]:
                    name = f"{act.name}_condition{i}"
                    self.add_condition(time_interval, fnode, name)
                    i += 1


class CPSETimepoints(CPSEBaseEngine):
    """
    Implementation of the CPSE Engine using an ordered list of timepoints.

    This class extends the `CPSEBaseEngine` and models a `SchedulingProblem` by defining
    an ordered list of timepoints. Each activity's start and end timepoints are mapped
    to distinct timepoints within this list.

    Attributes:
        timepoints (List[cp_model.IntVar]): An ordered list of timepoints in the model.
        assignment_matrix (Dict[Tuple[timing.Timepoint, int], List[cp_model.IntVar]]):
            A mapping from a timing to a list of boolean assignment variables.
            It represents a square binary matrix, where each row and column contains
            exactly one entry of `1` with all other entries being `0`, indicating the
            assignment of each timepoint to a unique problem timing.
        resources (Dict[FNode, List[List[cp_model.IntVar]]]): A dictionary tracking
            fluent values at each timepoint.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.timepoints: List[cp_model.IntVar] = []
        self.assignment_matrix: Dict[
            Tuple[timing.Timepoint, int], List[cp_model.IntVar]
        ] = {}
        self.resources: Dict[FNode, List[List[cp_model.IntVar]]] = {}
        self.parametric_fluent_assignments: Dict[
            FNode, List[Tuple[FNode, Tuple[Object], cp_model.IntVar]]
        ] = {}

    @property
    def name(self) -> str:
        return "CPSETimepoints"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = CPSEBaseEngine.supported_kind()
        supported_kind.set_initial_state("UNDEFINED_INITIAL_SYMBOLIC")
        supported_kind.set_initial_state("UNDEFINED_INITIAL_NUMERIC")
        supported_kind.set_effects_kind("FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_expression_duration("FLUENTS_IN_DURATIONS")
        return supported_kind

    @staticmethod
    def supports(problem_kind: ProblemKind) -> bool:
        return problem_kind <= CPSETimepoints.supported_kind()

    def check_if_supported_problem(self, problem: "up.model.AbstractProblem"):
        """
        Checks if the given problem is a supported instance of `SchedulingProblem`.
        Raises `NotImplementedError` if unsupported.

        Args:
            problem (up.model.AbstractProblem): The problem instance to validate.

        Raises:
            NotImplementedError: If the problem is not supported.
        """

        # TODO: add more checks

        if not isinstance(problem, SchedulingProblem):
            raise NotImplementedError(f"Problem of type {type(problem)} not supported.")
        problem: SchedulingProblem

        if not problem.discrete_time:
            raise NotImplementedError("Continuous time not supported.")

        fluents = collections.defaultdict(dict)
        contains_parameters = collections.defaultdict(dict)
        for timing, eff, activity in problem.all_effects():
            fluent = eff.fluent.fluent()
            contains_params = self._fluent_exp_contains_parameters(eff.fluent)
            if fluent in contains_parameters[timing] and (
                contains_parameters[timing][fluent] != contains_params
                or (
                    contains_parameters[timing][fluent]
                    and fluents[timing][fluent] != eff.fluent
                )
            ):
                raise NotImplementedError(
                    "Fluents with parameters cannot coexist with non-parametric fluents at the same timepoint."
                )
            fluents[timing][fluent] = eff.fluent
            contains_parameters[timing][fluent] = contains_params

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

        # TODO: avoid code duplication

        assert (
            not activity.duration.is_left_open()
            and not activity.duration.is_right_open()
        )
        lower = activity.duration.lower
        upper = activity.duration.upper
        if lower.is_int_constant() and upper.is_int_constant():
            lower = lower.int_constant_value()
            upper = upper.int_constant_value()
            if lower == upper:
                # FixedDuration
                start_var = self.model.new_int_var(
                    self.lower_bound, self.upper_bound, "start_" + activity.name
                )
                end_var = self.model.new_int_var(
                    self.lower_bound, self.upper_bound, "end_" + activity.name
                )
                duration_var = upper
            else:
                # ClosedDurationInterval
                start_var = lower
                end_var = upper
                duration_var = upper - lower
        else:
            start_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "start_" + activity.name
            )
            end_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "end_" + activity.name
            )
            duration_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "duration_" + activity.name
            )
            if lower == upper:
                # FixedDuration
                def constrain_duration():
                    self._set_fluent_vars_at_timepoint(tp_idx=-1)
                    duration_exp = self.fnode_to_value_or_variable(upper)
                    self.model.add(duration_var == duration_exp)

                self._postponed_constraints.append(constrain_duration)
            else:
                # ClosedDurationInterval
                def constrain_start_end():
                    self._set_fluent_vars_at_timepoint(tp_idx=-1)
                    lower_exp = self.fnode_to_value_or_variable(lower)
                    self.model.add(start_var == lower_exp)
                    upper_exp = self.fnode_to_value_or_variable(upper)
                    self.model.add(end_var == upper_exp)

                self._postponed_constraints.append(constrain_start_end)

        return start_var, duration_var, end_var

    def _fnode_timings(self, fnode: FNode) -> Iterable["timing.Timing"]:
        """
        Extracts all timings from the given FNode.

        Args:
            fnode (FNode): The expression tree to be traversed.

        Yields:
            timing.Timing: Each timing found within the FNode's tree.
        """

        stack = [fnode]
        while len(stack) > 0:
            fnode = stack.pop()
            if fnode.is_timing_exp():
                yield fnode.timing()

            if len(fnode.args) > 0:
                for arg in fnode.args:
                    stack.append(arg)

    def _collect_all_problem_timings(
        self, problem: SchedulingProblem
    ) -> List[Tuple["timing.Timepoint", int]]:
        """
        Collects all unique timepoints and their associated delays from the given scheduling problem.

        Args:
            problem (SchedulingProblem): The scheduling problem from which to extract timing information.

        Returns:
            List[Tuple["timing.Timepoint", int]]:
                A list of unique timepoints and their associated delays, represented as tuples
                of the form `(timepoint, delay)`.
        """

        problem_timings: Set[Tuple["timing.Timepoint", int]] = set()

        for activity in problem.activities:
            problem_timings.add((activity.start, 0))
            problem_timings.add((activity.end, 0))

        for fnode, activity in problem.all_constraints():
            for timing in self._fnode_timings(fnode):
                problem_timings.add((timing.timepoint, timing.delay))

        for time_interval, fnode, activity in problem.all_conditions():
            problem_timings.add(
                (time_interval.lower.timepoint, time_interval.lower.delay)
            )
            problem_timings.add(
                (time_interval.upper.timepoint, time_interval.upper.delay)
            )
            for timing in self._fnode_timings(fnode):
                problem_timings.add((timing.timepoint, timing.delay))

        for timing, effect, activity in problem.all_effects():
            problem_timings.add((timing.timepoint, timing.delay))
            if effect.is_conditional():
                for timing in self._fnode_timings(effect.condition):
                    problem_timings.add((timing.timepoint, timing.delay))

        return list(problem_timings)

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

    def get_all_fluent_expressions(self, problem: SchedulingProblem) -> Iterable[FNode]:
        """
        Retrieves all fluent expressions from the given scheduling problem.

        Args:
            problem (SchedulingProblem): The scheduling problem from which to extract fluent expressions.

        Yields:
            FNode: Each fluent expression.
        """

        # TODO: consider returning only the used parametric fluents

        for fluent in problem.fluents:
            if fluent.arity == 0:
                yield fluent()
                continue

            domains = []
            for arg in fluent.signature:
                domains.append(
                    self._get_compatible_objects(arg.type)
                    + self._get_compatible_parameters(arg.type)
                )

            for args in itertools.product(*domains):
                yield fluent(*args)

    def get_all_ground_fluent_expressions(
        self, problem: SchedulingProblem
    ) -> Iterable[FNode]:
        """
        Retrieves all ground fluent expressions from the given scheduling problem.

        Args:
            problem (SchedulingProblem): The scheduling problem from which to extract fluent expressions.

        Yields:
            FNode: Each ground fluent expression.
        """

        for fluent in problem.fluents:
            if fluent.arity == 0:
                yield fluent()
                continue

            domains = []
            for arg in fluent.signature:
                domains.append(self._get_compatible_objects(arg.type))

            for args in itertools.product(*domains):
                yield fluent(*args)

    def get_all_fluent_assignments(
        self, fluent_exps: List[FNode], params: List[Parameter]
    ) -> Iterable[List[Tuple[FNode, cp_model.IntVar]]]:
        """
        Generates all possible grounded assignments of parameters to fluent expressions.

        Args:
            fluent_exps (List[FNode]): A list of parametrized fluent expressions.
            params (List[Parameter]): A list of parameters for which compatible assignments
                will be generated.

        Returns:
            Iterable[List[Tuple[FNode, cp_model.IntVar]]]: An iterable where each element is a
                list of tuples. Each tuple contains:
                    - An `FNode` representing a grounded fluent expression.
                    - A `cp_model.IntVar` representing the corresponding model variable for the assignment.
        """

        domains = []
        for param in params:
            domains.append(self._get_compatible_objects(param.type))

        params_indexes = dict([(p, i) for i, p in enumerate(params)])

        for objs in itertools.product(*domains):
            ground_fluent_exps = []
            for fluent_exp in fluent_exps:
                fluent_objs = []
                for arg in fluent_exp.args:
                    if arg.is_parameter_exp():
                        fluent_objs.append(objs[params_indexes[arg.parameter()]])
                    else:
                        fluent_objs.append(arg.object())
                ground_fluent_exp = fluent_exp.fluent()(*fluent_objs)
                _, assignment_var = self.parametric_fluent_assignments[fluent_exp][
                    ground_fluent_exp
                ]
                ground_fluent_exps.append((ground_fluent_exp, assignment_var))
            yield ground_fluent_exps

    def all_parameters_used_in_fluents(
        self, problem: SchedulingProblem
    ) -> Set[Parameter]:
        """
        Identifies all parameters used in the fluent expressions of the given problem.

        Args:
            problem (SchedulingProblem): The scheduling problem containing fluent expressions.

        Returns:
            Set[Parameter]: A set of `Parameter` objects that are used in the fluent expressions.
        """

        all_parameters = set()
        for fluent_exp in self.get_all_fluent_expressions(problem):
            for arg in fluent_exp.args:
                if arg.is_parameter_exp():
                    all_parameters.add(arg.parameter())
        return all_parameters

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

    def _new_resource_var(self, fluent_exp: FNode, tp_idx: int) -> cp_model.IntVar:
        """
        Creates a new variable representing the value of a resource (fluent)
        at a specific timepoint.

        Args:
            fluent_exp (FNode): The fluent expression representing the resource.
            tp_idx (int): The timepoint index for which the variable is created.
                Must satisfy `-1 <= tp_idx < len(self.timepoints)`.

        Returns:
            cp_model.IntVar: A model variable representing the resource value at
                the specified timepoint.
        """

        assert fluent_exp.is_fluent_exp()
        assert -1 <= tp_idx < len(self.timepoints)
        self._int_var_counter += 1
        lb, ub = self._fluent_bounds[fluent_exp.fluent().name]
        resource_var = self.model.new_int_var(
            lb, ub, f"{fluent_exp}@{self.timepoints[tp_idx]}_{self._int_var_counter}"
        )
        self.resources[fluent_exp][tp_idx + 1].append(resource_var)
        return resource_var

    def _set_fluent_vars_at_timepoint(self, tp_idx: int):
        """
        Sets the model variables for all fluents at a specific timepoint.

        This method associates each fluent expression with its corresponding
        value at the given timepoint. The timepoint index must satisfy the
        constraint `-1 <= tp_idx < len(self.timepoints)`, where `-1` refers
        to the fluent's initial value.

        Args:
            tp_idx (int): The timepoint index used to associate each fluent with
            its corresponding value at the given timepoint. Must satisfy
            `-1 <= tp_idx < len(self.timepoints)`.
        """

        assert -1 <= tp_idx < len(self.timepoints)
        for fluent_exp in self.resources:
            self._model_vars[fluent_exp] = self.resources[fluent_exp][tp_idx + 1][-1]

    def timepoints_setup(self, problem: SchedulingProblem):
        """
        Defines the constraints that links each activity timepoint (start or end)
        to exactly one timepoint in the `timepoints` list.

        Args:
            problem (SchedulingProblem): The scheduling problem.
        """

        # TODO: try to use add_map_domain()

        problem_timings = self._collect_all_problem_timings(problem)
        self.timepoints = [
            self.model.new_int_var(self.lower_bound, self.upper_bound, f"timepoint{i}")
            for i in range(len(problem_timings))
        ]
        for i in range(len(self.timepoints) - 1):
            self.model.add(self.timepoints[i] <= self.timepoints[i + 1])

        self.assignment_matrix = {}
        for i in range(len(problem_timings)):
            timing = problem_timings[i]
            timepoint_var = self._model_vars[timing[0]]
            self.assignment_matrix[timing] = []
            for tp in self.timepoints:
                bool_var = self.model.new_bool_var(f"{timing}@{tp}")
                self.assignment_matrix[timing].append(bool_var)
                self.model.add(tp == (timepoint_var + timing[1])).only_enforce_if(
                    bool_var
                )

            # each problem timing is assigned to exactly one timepoint
            self.model.add_exactly_one(self.assignment_matrix[timing])

        # each timepoint is assigned to exactly one problem timing
        for i in range(len(self.timepoints)):
            self.model.add_exactly_one(
                bool_vars[i] for bool_vars in self.assignment_matrix.values()
            )

        # (num resources) x (num timepoints + 1)
        self.resources = {}
        for fluent_exp in self.get_all_ground_fluent_expressions(problem):
            lb, ub = self._fluent_bounds[fluent_exp.fluent().name]
            init_value_var = self.model.new_int_var(lb, ub, f"{fluent_exp}_init_value")
            self.resources[fluent_exp] = [[init_value_var]]
            for i in range(len(self.timepoints)):
                self.resources[fluent_exp].append([])

        self._set_fluent_vars_at_timepoint(tp_idx=-1)
        for fluent_exp in self.resources:
            if fluent_exp not in self._fluent_initial_value:
                # uninitialized fluent
                continue

            init_value = self._fluent_initial_value[fluent_exp]
            value_var = self.fnode_to_value_or_variable(init_value)
            self.model.add(self.resources[fluent_exp][0][0] == value_var)

    def add_constraints(self, problem: SchedulingProblem):
        """
        Adds all constraints from the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                constraints to be added to the model.
        """

        # TODO: avoid bool_var for the root node

        for fnode, activity in problem.all_constraints():
            if not self._fnode_contains_fluents(fnode):
                bool_var = self.add_constraint(fnode)
                self.model.add_bool_and([bool_var])
            else:
                all_fluent_exps = self.extract_all_fluent_exp_from_fnode(fnode)
                all_parameters = self.extract_all_params_from_fluent_exps(
                    all_fluent_exps
                )
                if len(all_parameters) == 0:
                    # for each timepoint, add a constraint defined on the fluents at that timepoint
                    for i in range(-1, len(self.timepoints)):
                        self._set_fluent_vars_at_timepoint(tp_idx=i)
                        bool_var = self.add_constraint(fnode)
                        self.model.add_bool_and([bool_var])

                else:
                    for ground_fluent_exps in self.get_all_fluent_assignments(
                        all_fluent_exps, all_parameters
                    ):
                        for i in range(-1, len(self.timepoints)):
                            self._set_fluent_vars_at_timepoint(tp_idx=i)
                            for fluent_exp, (ground_fluent_exp, assignment_var) in zip(
                                all_fluent_exps, ground_fluent_exps
                            ):
                                self._model_vars[fluent_exp] = self.resources[
                                    ground_fluent_exp
                                ][i + 1][-1]

                            bool_var = self.add_constraint(fnode)
                            self.model.add_bool_and([bool_var]).only_enforce_if(
                                [
                                    assignment_var
                                    for ground_fluent_exp, assignment_var in ground_fluent_exps
                                ]
                            )

    def add_parametric_fluents_constraints(self, problem: SchedulingProblem):
        """
        Adds constraints to map parametric fluents to ground fluents.

        Args:
            problem (SchedulingProblem): The scheduling problem that contains the fluents.
        """

        all_fluent_exps = self.get_all_fluent_expressions(problem)
        all_parameters = list(self.all_parameters_used_in_fluents(problem))

        params_objs_assignment = {}
        for param in all_parameters:
            params_objs_assignment[param] = {}
            for obj in self._type_objects_mapping[param.type]:
                bool_var = self.new_bool_var()
                self.model.add(
                    self._model_vars[param] == self.objects[obj]
                ).only_enforce_if(bool_var)
                params_objs_assignment[param][obj] = bool_var

        for fluent_exp in all_fluent_exps:
            if not self._fluent_exp_contains_parameters(fluent_exp):
                continue

            domains = []
            for arg in fluent_exp.args:
                if arg.is_parameter_exp():
                    domains.append(self._get_compatible_objects(arg.parameter().type))
                else:
                    assert arg.is_object_exp()
                    domains.append([arg.object()])

            fluent_assignment_vars = []
            for objs in itertools.product(*domains):
                params_assignment = []
                for arg, obj in zip(fluent_exp.args, objs):
                    if arg.is_parameter_exp():
                        params_assignment.append(
                            params_objs_assignment[arg.parameter()][obj]
                        )
                bool_var = self.new_bool_var()
                self.model.add_bool_and(params_assignment).only_enforce_if(bool_var)
                fluent_assignment_vars.append(bool_var)

                ground_fluent_exp = fluent_exp.fluent()(*objs)
                if fluent_exp not in self.parametric_fluent_assignments:
                    self.parametric_fluent_assignments[fluent_exp] = {}
                self.parametric_fluent_assignments[fluent_exp][ground_fluent_exp] = (
                    objs,
                    bool_var,
                )

            # each fluent expression with parameters is equal to exactly one ground fluent expression
            self.model.add_exactly_one(fluent_assignment_vars)

    def ground_fluent_exp(self, fluent_exp: FNode) -> Iterable[FNode]:
        """
        Yields the ground version of a fluent expression and its associated assignment variable, if applicable.

        This method checks if the given fluent expression contains parameters and, if so, generates
        all possible ground fluent expressions along with their associated assignment variables.
        If the fluent expression does not contain parameters, it is yielded as-is.

        Args:
            fluent_exp (FNode): The fluent expression to be grounded.

        Yields:
            Tuple[FNode, Union[cp_model.IntVar, None]]: A tuple containing the ground fluent expression
                and the associated boolean variable (or `None` if no variable is associated).
        """

        if self._fluent_exp_contains_parameters(fluent_exp):
            for ground_fluent_exp, (
                objs,
                bool_var,
            ) in self.parametric_fluent_assignments[fluent_exp].items():
                yield ground_fluent_exp, bool_var
        else:
            yield fluent_exp, None

    def _filter_fluent_effects(
        self, problem: SchedulingProblem, fluent_exp: FNode
    ) -> List[Tuple[timing.Timing, Effect]]:
        """
        Filters and returns the effects that apply specifically to the given fluent.

        Args:
            problem (SchedulingProblem): The scheduling problem.
            fluent_exp (FNode): The fluent expression whose relevant effects are to be identified.

        Returns:
            List[Tuple[timing.Timing, Effect]]: A list of effects that impact the specified fluent.
        """

        assert fluent_exp.is_fluent_exp()
        activity_effects = [
            (timing, eff)
            for act in problem.activities
            for timing, effects in act.effects.items()
            for eff in effects
            if eff.fluent == fluent_exp
        ]

        problem_effects = list(
            filter(
                lambda t_eff: t_eff[1].fluent == fluent_exp,
                problem.base_effects,
            )
        )

        return activity_effects + problem_effects

    def add_no_effect_constraints(
        self,
        fluent_effects: Dict[FNode, Dict["timing.Timing", Dict[str, List[Effect]]]],
    ):
        """
        Adds constraints to ensure that fluent values remain unchanged if no effects are
        applied at a given timepoint.

        This enforces the rule that if no effect modifies a fluent at a specific timepoint,
        its value will be equal to its value at the preceding timepoint.

        Args:
            fluent_effects (Dict[FNode, Dict["timing.Timing", Dict[str, List[Effect]]]]):
                A nested dictionary mapping fluent expressions to their effects at specific timepoints.
        """

        for fluent_exp in fluent_effects:
            for i, tp in enumerate(self.timepoints):
                timings = fluent_effects[fluent_exp].keys()
                tp_assignment_vars = list(
                    map(
                        lambda t: self.assignment_matrix[(t.timepoint, t.delay)][i],
                        timings,
                    )
                )

                no_effects_at_tp_var = self.new_bool_var()
                # not(assignment_vars[0] or ... or assignment_vars[-1]) => no_effects_at_tp_var
                self.model.add_bool_or(tp_assignment_vars + [no_effects_at_tp_var])

                prev_resource_value = self.resources[fluent_exp][i][-1]
                for resource_value in self.resources[fluent_exp][i + 1]:
                    self.model.add(
                        resource_value == prev_resource_value
                    ).only_enforce_if(no_effects_at_tp_var)
                    prev_resource_value = resource_value

    def add_assign_effect_constraints(
        self,
        fluent_effects: Dict[FNode, Dict["timing.Timing", Dict[str, List[Effect]]]],
        condition_vars: Dict[
            Effect, Union[cp_model.IntVar, cp_model._NotBooleanVariable]
        ],
    ):
        """
        Adds constraints to ensure that assignment effects do not occur simultaneously
        with any other effect on the same fluent.

        Args:
            fluent_effects (Dict[FNode, Dict["timing.Timing", Dict[str, List[Effect]]]]):
                A nested dictionary mapping fluent expressions to their effects at specific timepoints.
            condition_vars (Dict[Effect, Union[cp_model.IntVar, cp_model._NotBooleanVariable]]):
                A mapping of effects to their associated condition variables, used to enforce
                conditional effects.
        """

        constrained_timepoints = set()
        for fluent_exp in self.resources:
            for t1 in fluent_effects[fluent_exp]:
                for e1, fluent_assignment_var1 in (
                    fluent_effects[fluent_exp][t1]["conditional"]
                    + fluent_effects[fluent_exp][t1]["non_conditional"]
                ):
                    if not e1.is_assignment():
                        continue

                    if fluent_assignment_var1 is None:
                        fluent_assignment_var1 = True

                    for t2 in fluent_effects[fluent_exp]:
                        for e2, fluent_assignment_var2 in (
                            fluent_effects[fluent_exp][t2]["conditional"]
                            + fluent_effects[fluent_exp][t2]["non_conditional"]
                        ):
                            if t1 == t2 or (t1, t2) in constrained_timepoints:
                                continue

                            if fluent_assignment_var2 is None:
                                fluent_assignment_var2 = True

                            if e1.is_conditional():
                                condition_var1 = condition_vars[e1]
                            else:
                                condition_var1 = True

                            if e2.is_conditional():
                                condition_var2 = condition_vars[e2]
                            else:
                                condition_var2 = True

                            self.model.add(
                                (self._model_vars[t1.timepoint] + t1.delay)
                                != (self._model_vars[t2.timepoint] + t2.delay)
                            ).only_enforce_if(
                                [
                                    fluent_assignment_var1,
                                    fluent_assignment_var2,
                                    condition_var1,
                                    condition_var2,
                                ]
                            )

                            if all(
                                isinstance(v, bool)
                                for v in [
                                    fluent_assignment_var1,
                                    fluent_assignment_var2,
                                    condition_var1,
                                    condition_var2,
                                ]
                            ):
                                constrained_timepoints.add((t1, t2))

    def add_effect(
        self,
        timing: timing.Timing,
        fluent_exp: FNode,
        effect: Effect,
        fluent_idx: int,
        condition_vars: Dict[
            Effect, Union[cp_model.IntVar, cp_model._NotBooleanVariable]
        ],
        fluent_assignment_var: Union[cp_model.IntVar, None],
        value: Union[FNode, None] = None,
    ):
        """
        Adds an effect to the model.

        For each timepoint, if the effect is applied at that timepoint, the fluent value
        is updated to be equal to its value at the previous timepoint plus the effect.
        If the effect is an assignment, the fluent value at that timepoint is set directly
        to the effect value.

        Args:
            timing (timing.Timing): The specific timing at which the effect is applied.
            fluent_exp (FNode): The fluent expression to which the effect is applied.
            effect (Effect): The effect to be applied.
            fluent_idx (int): An index identifying the fluent variable to be updated.
            condition_vars (Dict[Effect, Union[cp_model.IntVar, cp_model._NotBooleanVariable]]):
                A mapping of effects to their associated condition variables, used to enforce
                conditional effects.
            fluent_assignment_var (Union[cp_model.IntVar, None]):
                If present, indicates that the fluent expression is mapped from a parametric
                fluent expression. This variable is used to determine if the parametric fluent
                expression is assigned to the corresponding ground fluent expression.
            value (Union[FNode, None], optional): The value that overrides the effect value.
                Defaults to None.
        """

        if effect.is_conditional():
            # TODO: effect conditions can use fluents ?
            if self._fnode_contains_fluents(effect.condition):
                raise NotImplementedError("Effect conditions must not include fluents.")
            condition_var = self.add_constraint(effect.condition)
            condition_vars[effect] = condition_var

        for i, bool_var in enumerate(
            self.assignment_matrix[(timing.timepoint, timing.delay)]
        ):
            self._set_fluent_vars_at_timepoint(tp_idx=i)
            resource_var = self.resources[fluent_exp][i + 1][fluent_idx]
            if value is None:
                value_var = self.fnode_to_value_or_variable(effect.value)
                if effect.is_decrease():
                    value_var = -value_var
            else:
                value_var = self.fnode_to_value_or_variable(value)

            if fluent_idx == 0:
                prev_var = self.resources[fluent_exp][i][-1]
            else:
                prev_var = self.resources[fluent_exp][i + 1][fluent_idx - 1]

            if effect.is_assignment():
                resource_equality = resource_var == value_var
            else:
                resource_equality = resource_var == (prev_var + value_var)

            if fluent_assignment_var is None:
                if effect.is_conditional():
                    self.model.add(resource_equality).only_enforce_if(
                        [bool_var, condition_var]
                    )
                    self.model.add(resource_var == prev_var).only_enforce_if(
                        [bool_var, condition_var.negated()]
                    )

                else:
                    self.model.add(resource_equality).only_enforce_if(bool_var)

            else:
                if effect.is_conditional():
                    self.model.add(resource_equality).only_enforce_if(
                        [bool_var, condition_var, fluent_assignment_var]
                    )
                    self.model.add(resource_var == prev_var).only_enforce_if(
                        [bool_var, condition_var, fluent_assignment_var.negated()]
                    )
                    self.model.add(resource_var == prev_var).only_enforce_if(
                        [bool_var, condition_var.negated(), fluent_assignment_var]
                    )
                    self.model.add(resource_var == prev_var).only_enforce_if(
                        [
                            bool_var,
                            condition_var.negated(),
                            fluent_assignment_var.negated(),
                        ]
                    )

                else:
                    self.model.add(resource_equality).only_enforce_if(
                        [bool_var, fluent_assignment_var]
                    )
                    self.model.add(resource_var == prev_var).only_enforce_if(
                        [bool_var, fluent_assignment_var.negated()]
                    )

    def add_effects(self, problem: SchedulingProblem):
        """
        Adds the problem effects to the model. For each fluent, if no effects are applied
        at a specific timepoint, the fluent retains its value from the previous timepoint.

        Args:
            problem (SchedulingProblem): The scheduling problem
        """

        # self.resources[fluent_exp] will contain for each timepoint:
        #   - one int var if there are non-conditional effects on the fluent
        #   - one int var for each conditional effect on the fluent

        self.timepoints_setup(problem)
        self.add_parametric_fluents_constraints(problem)

        # map each fluent to its effects
        fluent_effects: Dict[FNode, Dict["timing.Timing", Dict[str, List[Effect]]]] = {}
        for fluent_exp in self.resources:
            fluent_effects[fluent_exp] = {}

        for timing, eff, activity in problem.all_effects():
            for fluent_exp, assignment_var in self.ground_fluent_exp(eff.fluent):
                if timing not in fluent_effects[fluent_exp]:
                    fluent_effects[fluent_exp][timing] = {
                        "conditional": [],
                        "non_conditional": [],
                    }
                if eff.is_conditional():
                    fluent_effects[fluent_exp][timing]["conditional"].append(
                        (eff, assignment_var)
                    )
                else:
                    fluent_effects[fluent_exp][timing]["non_conditional"].append(
                        (eff, assignment_var)
                    )

        # map each fluent to its maximum number of (conditional) effects
        max_num_effects: Dict[FNode, int] = {}
        for fluent_exp in fluent_effects:
            max_num_effects[fluent_exp] = 0
            for timing in fluent_effects[fluent_exp]:
                max_num_effects[fluent_exp] = max(
                    max_num_effects[fluent_exp],
                    (
                        len(fluent_effects[fluent_exp][timing]["conditional"])
                        + min(
                            1,
                            len(fluent_effects[fluent_exp][timing]["non_conditional"]),
                        )
                    ),
                )

        # create the fluent variables
        for fluent_exp in max_num_effects:
            for i in range(len(self.timepoints)):
                if max_num_effects[fluent_exp] == 0:
                    # no effects
                    self._new_resource_var(fluent_exp, i)

                for _ in range(max_num_effects[fluent_exp]):
                    self._new_resource_var(fluent_exp, i)

        # add the effects as constraints to the model
        # sum the effect values of non-conditional effects
        condition_vars: Dict[
            Effect, Union[cp_model.IntVar, cp_model._NotBooleanVariable]
        ] = {}
        for fluent_exp in fluent_effects:
            for timing in fluent_effects[fluent_exp]:
                sum_inc_dec_effects = 0
                idx = 0
                for eff, fluent_assignment_var in fluent_effects[fluent_exp][timing][
                    "non_conditional"
                ]:
                    if eff.is_assignment():
                        assert (
                            len(fluent_effects[fluent_exp][timing]["non_conditional"])
                            == 1
                        ), "Multiple effects on the same fluent at the same timepoint are not allowed when an assignment effect is present."
                        self.add_effect(
                            timing,
                            fluent_exp,
                            eff,
                            idx,
                            condition_vars,
                            fluent_assignment_var,
                        )
                        continue

                    # sum values of different effects that occur at the same timepoint
                    if eff.is_increase():
                        sum_inc_dec_effects = Plus(sum_inc_dec_effects, eff.value)
                    else:
                        sum_inc_dec_effects = Minus(sum_inc_dec_effects, eff.value)

                if sum_inc_dec_effects != 0:
                    self.add_effect(
                        timing,
                        fluent_exp,
                        eff,
                        idx,
                        condition_vars,
                        fluent_assignment_var,
                        value=sum_inc_dec_effects,
                    )

                idx = min(1, len(fluent_effects[fluent_exp][timing]["non_conditional"]))
                for eff, fluent_assignment_var in fluent_effects[fluent_exp][timing][
                    "conditional"
                ]:
                    self.add_effect(
                        timing,
                        fluent_exp,
                        eff,
                        idx,
                        condition_vars,
                        fluent_assignment_var,
                    )
                    idx += 1

                # propagate the value to the last variable when the number of effects
                # is less than the maximum one
                if idx < max_num_effects[fluent_exp]:
                    for i, bool_var in enumerate(
                        self.assignment_matrix[(timing.timepoint, timing.delay)]
                    ):
                        resource_var = self.resources[fluent_exp][i + 1][-1]
                        if idx == 0:
                            prev_var = self.resources[fluent_exp][i][-1]
                        else:
                            prev_var = self.resources[fluent_exp][i + 1][idx - 1]

                        self.model.add(resource_var == prev_var).only_enforce_if(
                            bool_var
                        )

        self.add_no_effect_constraints(fluent_effects)
        self.add_assign_effect_constraints(fluent_effects, condition_vars)

    def add_condition(
        self,
        time_interval: timing.TimeInterval,
        fnode: FNode,
    ):
        """
        Adds a condition to the model, enforcing that it is satisfied within a specified time interval.

        Args:
            time_interval (timing.TimeInterval): The time interval during which the condition
                must be satisfied.
            fnode (FNode): The FNode representing the condition to be added as a constraint.
        """

        start_delay = 0
        if time_interval.lower.delay != 0:
            start_delay += time_interval.lower.delay
        if time_interval.is_left_open():
            start_delay += 1
        start = self._model_vars[time_interval.lower.timepoint] + start_delay

        end_delay = 0
        if time_interval.upper.delay != 0:
            end_delay += time_interval.upper.delay
        if time_interval.is_right_open():
            end_delay -= 1
        end = self._model_vars[time_interval.upper.timepoint] + end_delay

        # if there are no fluents in the condition, a constraint should be added
        if not self._fnode_contains_fluents(fnode):
            constraint_var = self.add_constraint(fnode)
            self.model.add(start <= end).only_enforce_if(constraint_var)
            self.model.add(start > end).only_enforce_if(constraint_var.negated())
            return

        all_fluent_exps = self.extract_all_fluent_exp_from_fnode(fnode)
        all_parameters = self.extract_all_params_from_fluent_exps(all_fluent_exps)
        all_fluent_assignments = [None]
        if len(all_parameters) > 0:
            all_fluent_assignments = self.get_all_fluent_assignments(
                all_fluent_exps, all_parameters
            )

        for ground_fluent_exps in all_fluent_assignments:
            # for each timepoint, add a constraint defined on the fluents at that timepoint
            # and enforce:
            #   (start <= timepoint <= end) => constraint
            for i in range(len(self.timepoints)):
                self._set_fluent_vars_at_timepoint(tp_idx=i)
                if ground_fluent_exps is not None:
                    for fluent_exp, (ground_fluent_exp, assignment_var) in zip(
                        all_fluent_exps, ground_fluent_exps
                    ):
                        self._model_vars[fluent_exp] = self.resources[
                            ground_fluent_exp
                        ][i + 1][-1]
                constraint_var = self.add_constraint(fnode)

                tp_GE_start_key = f"{self.timepoints[i].name} >= {start.name}"
                if tp_GE_start_key not in self._variables_cache:
                    tp_GE_start = self.new_bool_var()
                    self.model.add(self.timepoints[i] >= start).only_enforce_if(
                        tp_GE_start
                    )
                    self.model.add(self.timepoints[i] < start).only_enforce_if(
                        tp_GE_start.negated()
                    )
                    self._variables_cache[tp_GE_start_key] = tp_GE_start
                tp_GE_start = self._variables_cache[tp_GE_start_key]

                tp_LE_end_key = f"{self.timepoints[i].name} <= {end.name}"
                if tp_LE_end_key not in self._variables_cache:
                    tp_LE_end = self.new_bool_var()
                    self.model.add(self.timepoints[i] <= end).only_enforce_if(tp_LE_end)
                    self.model.add(self.timepoints[i] > end).only_enforce_if(
                        tp_LE_end.negated()
                    )
                    self._variables_cache[tp_LE_end_key] = tp_LE_end
                tp_LE_end = self._variables_cache[tp_LE_end_key]

                # if the next timepoint value is equal then the constraint should not be enforced
                # because fluent values should be taken from the last timepoint with that value
                if i + 1 >= len(self.timepoints):  # last timepoint
                    # (tp_GE_start and tp_LE_end) => constraint
                    if ground_fluent_exps is None:
                        self.model.add_bool_or(
                            [tp_GE_start.negated(), tp_LE_end.negated(), constraint_var]
                        )
                    else:
                        self.model.add_bool_or(
                            [
                                assignment_var.negated()
                                for ground_fluent_exp, assignment_var in ground_fluent_exps
                            ]
                            + [
                                tp_GE_start.negated(),
                                tp_LE_end.negated(),
                                constraint_var,
                            ]
                        )
                else:
                    next_timepoint_is_different_key = (
                        f"{self.timepoints[i + 1].name} != {self.timepoints[i].name}"
                    )
                    if next_timepoint_is_different_key not in self._variables_cache:
                        next_timepoint_is_different = self.new_bool_var()
                        self.model.add(
                            self.timepoints[i + 1] != self.timepoints[i]
                        ).only_enforce_if(next_timepoint_is_different)
                        self.model.add(
                            self.timepoints[i + 1] == self.timepoints[i]
                        ).only_enforce_if(next_timepoint_is_different.negated())
                        self._variables_cache[next_timepoint_is_different_key] = (
                            next_timepoint_is_different
                        )
                    next_timepoint_is_different = self._variables_cache[
                        next_timepoint_is_different_key
                    ]

                    # (tp_GE_start and tp_LE_end) => constraint
                    if ground_fluent_exps is None:
                        self.model.add_bool_or(
                            [tp_GE_start.negated(), tp_LE_end.negated(), constraint_var]
                        ).only_enforce_if(next_timepoint_is_different)
                    else:
                        self.model.add_bool_or(
                            [
                                assignment_var.negated()
                                for ground_fluent_exp, assignment_var in ground_fluent_exps
                            ]
                            + [
                                tp_GE_start.negated(),
                                tp_LE_end.negated(),
                                constraint_var,
                            ]
                        ).only_enforce_if(next_timepoint_is_different)

    def add_conditions(self, problem: SchedulingProblem):
        """
        Adds the conditions of the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                conditions to be added to the model.
        """

        for time_interval, fnode, activity in problem.all_conditions():
            self.add_condition(time_interval, fnode)
