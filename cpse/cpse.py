#!/usr/bin/env python3
"""Unified Planning Integration for OR-Tools CP-SAT Model"""
from typing import IO, Callable, Optional, List, Tuple, Union, Dict
from abc import abstractmethod
import collections
import operator
import warnings
import traceback

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
)
from unified_planning.model.scheduling import SchedulingProblem, Activity
from unified_planning.model.metrics import MinimizeMakespan
from unified_planning.plans import Schedule

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

# TODO: remove unused keys
_OPERATOR_MAP = {
    OperatorKind.AND: None,
    OperatorKind.OR: None,
    OperatorKind.NOT: None,
    OperatorKind.IMPLIES: None,
    OperatorKind.IFF: None,
    # OperatorKind.EXISTS: None,
    # OperatorKind.FORALL: None,
    # OperatorKind.FLUENT_EXP: None,
    OperatorKind.PARAM_EXP: None,
    # OperatorKind.VARIABLE_EXP: None,
    # OperatorKind.OBJECT_EXP: None,
    OperatorKind.TIMING_EXP: None,
    OperatorKind.BOOL_CONSTANT: None,
    OperatorKind.INT_CONSTANT: None,
    # OperatorKind.REAL_CONSTANT: None,
    OperatorKind.PLUS: operator.add,
    OperatorKind.MINUS: operator.sub,
    OperatorKind.TIMES: operator.mul,
    OperatorKind.DIV: operator.floordiv,
    OperatorKind.LE: operator.le,
    OperatorKind.LT: operator.lt,
    OperatorKind.EQUALS: operator.eq,
    # OperatorKind.ALWAYS: None,
    # OperatorKind.SOMETIME: None,
    # OperatorKind.SOMETIME_BEFORE: None,
    # OperatorKind.SOMETIME_AFTER: None,
    # OperatorKind.AT_MOST_ONCE: None,
    # OperatorKind.DOT: None,
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
        activities (Dict[str, activity_type]): A dictionary of activities keyed by name.
        model_vars (Dict[Union[timing.Timepoint, Parameter, Fluent], cp_model.IntVar]): A mapping
            of timepoints, parameters and fluents to the corresponding integer variables in the model.
        fluent_capacity (Dict[str, int]): A dictionary defining the maximum capacity for each fluent.
        fluent_initial_value (Dict[str, int]): A dictionary defining the initial value for each fluent.
        bool_var_counter (int): A counter for generating anonymous boolean variables.
        int_var_counter (int): A counter for generating anonymous integer variables.
    """

    def __init__(self, **kwargs):
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)

        self.lower_bound: int = kwargs.get("lower_bound", 0)
        self.upper_bound: int = kwargs.get("upper_bound", cp_model.INT32_MAX)
        self.model = cp_model.CpModel()

        self.activities: Dict[str, activity_type] = {}
        self.model_vars: Dict[
            Union[timing.Timepoint, Parameter, Fluent], cp_model.IntVar
        ] = {}
        self.fluent_capacity: Dict[str, int] = {}
        self.fluent_initial_value: Dict[str, int] = {}

        self.bool_var_counter: int = -1
        self.int_var_counter: int = -1

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
        supported_kind.set_time("DISCRETE_TIME")
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
        supported_kind.set_parameters("BOOL_ACTION_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_parameters("UNBOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_quality_metrics("MAKESPAN")
        return supported_kind

    def new_bool_var(self) -> cp_model.IntVar:
        """
        Adds a new anonymous boolean variable to the model.

        Returns:
            cp_model.IntVar: A new integer variable bounded in [0,1].
        """

        self.bool_var_counter += 1
        return self.model.new_bool_var(f"bool{self.bool_var_counter}")

    def new_int_var(self) -> cp_model.IntVar:
        """
        Adds a new anonymous integer variable to the model.

        Returns:
            cp_model.IntVar: A new integer variable.
        """

        self.int_var_counter += 1
        return self.model.new_int_var(
            self.lower_bound, self.upper_bound, f"int{self.int_var_counter}"
        )

    def _convert_timing_to_linear_expr(self, t: timing.Timing) -> cp_model.LinearExprT:
        """
        Converts a `timing.Timing` variable to a `cp_model.LinearExprT`.

        Args:
            t (timing.Timing): The timing variable.

        Returns:
            cp_model.LinearExprT: The corresponding linear expression in the model.
        """

        assert t.timepoint in self.model_vars
        assert isinstance(t.delay, int)
        return cp_model.LinearExpr.sum([self.model_vars[t.timepoint], t.delay])

    def add_parameters(self, parameters: List[Parameter]):
        """
        Adds the parameters to the model as boolean or integer variables.

        Args:
            parameters (List[Parameter]): A list of parameters to be added to the model.
        """

        for param in parameters:
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
            else:
                raise NotImplementedError(f"Parameter type {param.type} not supported.")

            assert param not in self.model_vars
            self.model_vars[param] = var

    def add_activity(self, activity: Activity):
        """
        Adds an activity to the model. Each activity is modeled using an interval.

        Args:
            activity (Activity): The activity to be added to the model.
        """

        # assume FixedDuration or ClosedDurationInterval
        if not (
            activity.duration.lower.is_int_constant()
            and activity.duration.upper.is_int_constant()
            and not activity.duration.is_left_open()
            and not activity.duration.is_right_open()
        ):
            # TODO: support bounds defined using a parameter and STATIC_FLUENTS_IN_DURATION ?
            raise NotImplementedError(
                "Activity duration bounds must be closed and of integer type."
            )

        lower = activity.duration.lower.int_constant_value()
        upper = activity.duration.upper.int_constant_value()

        if lower == upper:
            # FixedDuration
            start_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "start_" + activity.name
            )
            end_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "end_" + activity.name
            )
            duration = upper
        else:
            # ClosedDurationInterval
            start_var = lower
            end_var = upper
            duration = upper - lower

        interval_var = self.model.new_interval_var(
            start_var, duration, end_var, activity.name
        )
        self.activities[activity.name] = activity_type(
            start=start_var, end=end_var, interval=interval_var, up_activity=activity
        )
        self.model_vars[activity.start] = start_var
        self.model_vars[activity.end] = end_var
        self._variables_cache[f"interval [{activity.start}, {activity.end}]"] = (
            interval_var
        )

    def add_constraint(
        self, fnode: FNode, cache_enabled=True
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

        Returns:
            Union[cp_model.IntVar, cp_model._NotBooleanVariable]:
                A boolean variable (or its negation) representing the constraint.
        """

        # TODO(cpse-timepoints): cannot use cache on fnodes with fluents

        stack = [(fnode, False)]
        results = []

        while len(stack) > 0:
            fnode, processed = stack.pop()

            # check if fnode cached
            if cache_enabled and not processed and repr(fnode) in self._variables_cache:
                results.append(self._variables_cache[repr(fnode)])

            elif fnode.is_parameter_exp():
                results.append(self.model_vars[fnode.parameter()])

            elif fnode.is_timing_exp():
                results.append(self._convert_timing_to_linear_expr(fnode.timing()))

            elif fnode.is_fluent_exp():
                if fnode.fluent() not in self.model_vars:
                    raise NotImplementedError(
                        f"Node type {fnode.node_type} not supported."
                    )
                results.append(self.model_vars[fnode.fluent()])

            elif fnode.node_type in [
                OperatorKind.BOOL_CONSTANT,
                OperatorKind.INT_CONSTANT,
            ]:
                results.append(fnode.constant_value())

            elif fnode.node_type in [
                OperatorKind.PLUS,
                OperatorKind.MINUS,
                OperatorKind.TIMES,
                OperatorKind.DIV,
            ]:
                if not processed:
                    stack.append((fnode, True))
                    for arg in fnode.args:
                        stack.append((arg, False))
                else:
                    op = _OPERATOR_MAP[fnode.node_type]
                    args = [results.pop() for arg in fnode.args]
                    results.append(op(*args))

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

            elif fnode.node_type in [
                OperatorKind.LE,
                OperatorKind.LT,
                OperatorKind.EQUALS,
            ]:
                if not processed:
                    assert len(fnode.args) == 2
                    stack.append((fnode, True))
                    for arg in fnode.args:
                        stack.append((arg, False))
                    continue

                args = [results.pop() for arg in fnode.args]
                op = _OPERATOR_MAP[fnode.node_type]
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

    def check_if_supported_problem(self, problem: "up.model.AbstractProblem"):
        """
        Checks if the given problem is a supported instance of `SchedulingProblem`.
        Raises `NotImplementedError` if unsupported.

        Parameters:
            problem (up.model.AbstractProblem): The problem instance to validate.

        Raises:
            NotImplementedError: If the problem is not supported.
        """

        if not isinstance(problem, SchedulingProblem):
            raise NotImplementedError(f"Problem of type {type(problem)} not supported.")
        problem: SchedulingProblem

        if not problem.discrete_time:
            raise NotImplementedError("Continuous time not supported.")
        if len(problem.user_types) > 0:
            raise NotImplementedError("User types not supported.")
        if len(problem.all_objects) > 0:
            raise NotImplementedError("Objects not supported.")

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

            # map each fluent to its maximum capacity and initial value
            for f in problem.fluents:
                self.fluent_capacity[f.name] = problem.fluents_defaults[
                    f
                ].int_constant_value()
                self.fluent_initial_value[f.name] = problem.fluents_defaults[
                    f
                ].int_constant_value()

            # override initial values when an explicit one is defined
            for f in problem.explicit_initial_values:
                self.fluent_initial_value[f.fluent().name] = (
                    problem.explicit_initial_values[f].int_constant_value()
                )

            # add the parameters of the problem to the model
            self.add_parameters(problem.base_variables)
            # add the parameters of the activities to the model
            for up_activity in problem.activities:
                self.add_parameters(up_activity.parameters)

            # add the activities to the model
            for up_activity in problem.activities:
                self.add_activity(up_activity)

            # define the makespan variable
            makespan_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "makespan"
            )
            self.model.add_max_equality(
                makespan_var, [a.end for a in self.activities.values()]
            )

            # add global start and end to the model variables
            self.model_vars[timing.GlobalStartTiming(delay=0).timepoint] = 0
            self.model_vars[timing.GlobalEndTiming().timepoint] = makespan_var

            # add problem-specific and activity-related effects to the model
            self.add_effects(problem)
            # add problem-specific and activity-related constraints to the model
            self.add_constraints(problem)
            # add problem-specific and activity-related conditions to the model
            self.add_conditions(problem)

            # add quality metrics to the model
            self.add_quality_metrics(problem, makespan_var)

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
                for up_var, cp_var in self.model_vars.items():
                    assignment[up_var] = solver.value(cp_var)
                    # map a boolean parameter to its boolean value rather than the
                    # integer value returned by the solver
                    if isinstance(up_var, Parameter) and up_var.type.is_bool_type():
                        assert solver.value(cp_var) in [0, 1]
                        assignment[up_var] = solver.value(cp_var) == 1

                plan = Schedule(problem.activities, assignment, problem.environment)

                return PlanGenerationResult(
                    (
                        PlanGenerationResultStatus.SOLVED_OPTIMALLY
                        if status == cp_model.OPTIMAL
                        else PlanGenerationResultStatus.SOLVED_SATISFICING
                    ),
                    plan,
                    engine_name=self.name,
                    log_messages=None,
                    metrics=metrics,
                )
            else:
                return PlanGenerationResult(
                    PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
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
    def supported_kind() -> ProblemKind:
        supported_kind = CPSEBaseEngine.supported_kind()
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_time("INTERMEDIATE_CONDITIONS_AND_EFFECTS")
        supported_kind.set_time("EXTERNAL_CONDITIONS_AND_EFFECTS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        return supported_kind

    @staticmethod
    def supports(problem_kind: ProblemKind) -> bool:
        return problem_kind <= CPSE.supported_kind()

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

        activities_effects = [
            (timing, eff)
            for act in problem.activities
            for timing, effects in act.effects.items()
            for eff in effects
        ]

        # map each fluent to its effects, adjusting values based on increase/decrease types
        fluent_effects: Dict[
            str, List[Tuple["timing.Timing", int, cp_model.IntVar]]
        ] = {}
        for timing, eff in activities_effects + problem.base_effects:
            eff: Effect
            fluent_name = eff.fluent.fluent().name
            if not eff.value.is_int_constant():
                raise NotImplementedError(
                    "Only constant integer effect values are supported."
                )
            value = eff.value.int_constant_value()
            if value == 0:  # if value is 0, the effect is ignored
                continue

            if fluent_name not in fluent_effects:
                fluent_effects[fluent_name] = []

            if eff.is_increase():
                pass
            elif eff.is_decrease():
                value = -value
            else:
                raise NotImplementedError(f"Effect kind {eff.kind} not supported.")

            if eff.is_conditional():
                bool_var = self.add_constraint(eff.condition)
                fluent_effects[fluent_name].append((timing, value, bool_var))
            else:
                fluent_effects[fluent_name].append((timing, value, True))

        for fluent_name in fluent_effects:
            times = [0]
            values = [self.fluent_initial_value[fluent_name]]
            actives = [True]
            for timing, value, active in fluent_effects[fluent_name]:
                times.append(self._convert_timing_to_linear_expr(timing))
                values.append(value)
                actives.append(active)

            self.model.add_reservoir_constraint_with_active(
                times, values, actives, 0, self.fluent_capacity[fluent_name]
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

        delay_start = 0
        if time_interval.lower.delay != 0:
            assert isinstance(time_interval.lower.delay, int)
            delay_start += time_interval.lower.delay
        if time_interval.is_left_open():
            delay_start += 1
        start = self.model_vars[time_interval.lower.timepoint] + delay_start

        # add 1 to end because `add_cumulative` enforces the constraint for t in [start, end),
        # but we want the constraint to be enforced also at the end
        delay_end = 1
        if time_interval.upper.delay != 0:
            assert isinstance(time_interval.upper.delay, int)
            delay_end += time_interval.upper.delay
        if time_interval.is_right_open():
            delay_end -= 1
        end = self.model_vars[time_interval.upper.timepoint] + delay_end

        interval_key = f"interval [{time_interval.lower.timepoint} + {delay_start}, {time_interval.upper.timepoint} + {delay_end}]"
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
        assignment_matrix (Dict[timing.Timepoint, List[cp_model.IntVar]]): A mapping from
            each timepoint to a list of boolean assignment variables.
            It represents a square binary matrix, where each row and column contains
            exactly one entry of `1` with all other entries being `0`, indicating the
            assignment of each timepoint to a unique activity timepoint.
        resources (Dict[str, List[cp_model.IntVar]]): A dictionary tracking resource value
            at each timepoint.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.timepoints: List[cp_model.IntVar] = []
        self.assignment_matrix: Dict[timing.Timepoint, List[cp_model.IntVar]] = {}
        self.resources: Dict[str, List[cp_model.IntVar]] = {}

    @property
    def name(self) -> str:
        return "CPSETimepoints"

    @staticmethod
    def supports(problem_kind: ProblemKind) -> bool:
        return problem_kind <= CPSEBaseEngine.supported_kind()

    def timepoints_setup(self, problem: SchedulingProblem):
        """
        Defines the constraints that links each activity timepoint (start or end)
        to exactly one timepoint in the `timepoints` list.

        Args:
            problem (SchedulingProblem): The scheduling problem.
        """

        self.timepoints = [
            self.model.new_int_var(self.lower_bound, self.upper_bound, f"timepoint{i}")
            for i in range(len(problem.activities) * 2)
        ]
        for i in range(len(self.timepoints) - 1):
            self.model.add(self.timepoints[i] <= self.timepoints[i + 1])

        activity_timepoints: List[Tuple[timing.Timepoint, cp_model.IntVar]] = []
        for activity_name in sorted(self.activities.keys()):
            activity = self.activities[activity_name]
            activity_timepoints.append((activity.up_activity.start, activity.start))
            activity_timepoints.append((activity.up_activity.end, activity.end))
        self.assignment_matrix = {}
        for i in range(len(activity_timepoints)):
            up_tp_i, activity_timepoint_i = activity_timepoints[i]
            # TODO: try to use add_map_domain()
            self.assignment_matrix[up_tp_i] = []
            for tp in self.timepoints:
                bool_var = self.model.new_bool_var(f"{activity_timepoint_i}@{tp}")
                self.assignment_matrix[up_tp_i].append(bool_var)
                self.model.add(tp == activity_timepoint_i).only_enforce_if(bool_var)

            # each activity timepoint is assigned to exactly one timepoint
            self.model.add_exactly_one(self.assignment_matrix[up_tp_i])

        # each timepoint is assigned to exactly one activity timepoint
        for i in range(len(self.timepoints)):
            self.model.add_exactly_one(
                bool_vars[i] for bool_vars in self.assignment_matrix.values()
            )

        # (num resources) x (num timepoints + 1)
        self.resources = {}
        for fluent in problem.fluents:
            init_value = self.model.new_constant(self.fluent_initial_value[fluent.name])
            self.resources[fluent.name] = [init_value]
            for i in range(len(self.timepoints)):
                resource_var = self.model.new_int_var(
                    0,
                    self.fluent_capacity[fluent.name],
                    f"{fluent.name}@{self.timepoints[i]}",
                )
                self.resources[fluent.name].append(resource_var)

    def add_constraints(self, problem: SchedulingProblem):
        """
        Adds all constraints from the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                constraints to be added to the model.
        """

        # TODO: avoid bool_var for the root node

        constraints = problem.base_constraints + [
            fnode for act in problem.activities for fnode in act.constraints
        ]

        for fnode in constraints:
            if not self._fnode_contains_fluents(fnode):
                bool_var = self.add_constraint(fnode)
                self.model.add_bool_and([bool_var])
            else:
                # for each timepoint, add a constraint defined on the fluents at that timepoint
                for i in range(len(self.timepoints)):
                    for fluent in problem.fluents:
                        self.model_vars[fluent] = self.resources[fluent.name][i + 1]
                    bool_var = self.add_constraint(fnode, cache_enabled=False)
                    self.model.add_bool_and([bool_var])

    def _filter_fluent_effects(
        self, problem: SchedulingProblem, fluent_name: str
    ) -> List[Tuple[timing.Timing, Effect]]:
        """
        Filters and returns the effects that apply specifically to the given fluent.

        Args:
            problem (SchedulingProblem): The scheduling problem.
            fluent_name (str): The fluent whose relevant effects are to be identified.

        Returns:
            List[Tuple[timing.Timing, Effect]]: A list of effects that impact the specified fluent.
        """

        activity_effects = [
            (timing, eff)
            for act in problem.activities
            for timing, effects in act.effects.items()
            for eff in effects
            if eff.fluent.fluent().name == fluent_name
        ]

        problem_effects = list(
            filter(
                lambda t_eff: t_eff[1].fluent.fluent().name == fluent_name,
                problem.base_effects,
            )
        )

        return activity_effects + problem_effects

    def add_assign_effect_constraints(self, problem: SchedulingProblem):
        """
        Adds constraints to ensure that assignment effects do not occur simultaneously
        with any other effect on the same fluent.

        Args:
            problem (SchedulingProblem): The scheduling problem.
        """

        visited_timepoints = set()
        for fluent in problem.fluents:
            fluent_effects = self._filter_fluent_effects(problem, fluent.name)
            assign_effects = list(
                filter(lambda t_eff: t_eff[1].is_assignment(), fluent_effects)
            )
            increase_decrease_effects = list(
                filter(lambda t_eff: not t_eff[1].is_assignment(), fluent_effects)
            )

            activity_timepoints = list(
                set(  # remove duplicates
                    map(
                        lambda t_eff: self.model_vars[t_eff[0].timepoint],
                        assign_effects,
                    )
                )
            )

            # TODO: use pairwise inequalities ?
            self.model.add_all_different(activity_timepoints)

            # add timepoints to visited_timepoints
            for i in range(len(activity_timepoints)):
                for j in range(i + 1, len(activity_timepoints)):
                    t1 = activity_timepoints[i]
                    t2 = activity_timepoints[j]
                    visited_timepoints.add((str(t1), str(t2)))
                    visited_timepoints.add((str(t2), str(t1)))

            for t1, e1 in assign_effects:
                for t2, e2 in increase_decrease_effects:
                    if t1 == t2 or (str(t1), str(t2)) in visited_timepoints:
                        continue

                    visited_timepoints.add((str(t1), str(t2)))
                    visited_timepoints.add((str(t2), str(t1)))

                    self.model.add(
                        self.model_vars[t1.timepoint] != self.model_vars[t2.timepoint]
                    )

    def add_effect(self, timing: timing.Timing, effect: Effect, value: int):
        """
        Adds an effect to the model.

        For each timepoint, if the effect is applied at that timepoint, the fluent value
        is updated to be equal to its value at the previous timepoint plus the effect.
        If the effect is an assignment, the fluent value at that timepoint is set directly
        to the effect value.

        Args:
            timing (timing.Timing): The specific timing at which the effect is applied.
            effect (Effect): The effect to apply.
            value (int): The value associated with the effect. This may represent the sum
                of increase/decrease effects or the exact value of the effect.
        """

        fluent_name = effect.fluent.fluent().name
        if timing.delay != 0:
            raise NotImplementedError("Timing delay not supported.")

        # TODO: support global timings and delays

        for i, bool_var in enumerate(self.assignment_matrix[timing.timepoint]):
            if effect.is_assignment():
                self.model.add(
                    self.resources[fluent_name][i + 1] == value
                ).only_enforce_if(bool_var)

            else:
                self.model.add(
                    self.resources[fluent_name][i + 1]
                    == (self.resources[fluent_name][i] + value)
                ).only_enforce_if(bool_var)

    def add_effects(self, problem: SchedulingProblem):
        """
        Adds the problem effects to the model. For each fluent, if an effect is not applied
        at a specific timepoint, the fluent retains its value from the previous timepoint.

        Args:
            problem (SchedulingProblem): The scheduling problem
        """

        # TODO: model conditional effects
        # conditional effects:
        # model value as List[int_var], where len is num effects at that timepoint on that fluent
        # for each effect:
        #   condition => value = prev_value + v
        #   not condition => value = prev_value

        self.timepoints_setup(problem)

        activities_effects = [
            (timing, eff)
            for act in problem.activities
            for timing, effects in act.effects.items()
            for eff in effects
        ]

        # sum values of different effects that occur at the same timepoint
        fluent_inc_dec_effects: Dict[str, Dict["timing.Timing", Effect]] = {}
        for timing, eff in activities_effects + problem.base_effects:
            eff: Effect
            fluent_name = eff.fluent.fluent().name
            value = eff.value.int_constant_value()
            if eff.is_increase():
                pass
            elif eff.is_decrease():
                value = -value
            else:
                self.add_effect(timing, eff, value)

            if fluent_name not in fluent_inc_dec_effects:
                fluent_inc_dec_effects[fluent_name] = {}

            if timing not in fluent_inc_dec_effects[fluent_name]:
                fluent_inc_dec_effects[fluent_name][timing] = [eff, value]
            else:
                # sum values of different effects that occur at the same timepoint
                fluent_inc_dec_effects[fluent_name][timing][1] += value

        for fluent_name in fluent_inc_dec_effects:
            for timing, (eff, value) in fluent_inc_dec_effects[fluent_name].items():
                self.add_effect(timing, eff, value)

        # for each resource
        #   for each timepoint
        #       if no activities have effects on that resource at that timepoint
        #           resource@tp(i) == resource@tp(i-1)
        for fluent_name in self.resources:
            fluent_effects = self._filter_fluent_effects(problem, fluent_name)
            for i, tp in enumerate(self.timepoints):
                assignment_vars = list(
                    map(
                        lambda t_eff: self.assignment_matrix[t_eff[0].timepoint][i],
                        fluent_effects,
                    )
                )
                bool_var = self.new_bool_var()
                self.model.add(
                    self.resources[fluent_name][i + 1] == self.resources[fluent_name][i]
                ).only_enforce_if(bool_var)
                # not(assignment_vars[0] or ... or assignment_vars[-1]) => bool_var
                self.model.add_bool_or(assignment_vars + [bool_var])

        self.add_assign_effect_constraints(problem)

    def _fnode_contains_fluents(self, fnode: FNode) -> bool:
        """
        Checks if the specified FNode contains any fluent.

        Returns:
            bool: `True` if a fluent is found within the given `fnode`, otherwise `False`.
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

    def add_condition(
        self,
        time_interval: timing.TimeInterval,
        fnode: FNode,
        problem: SchedulingProblem,
    ):
        """
        Adds a condition to the model, enforcing that it is satisfied within a specified time interval.

        Args:
            time_interval (timing.TimeInterval): The time interval during which the condition
                must be satisfied.
            fnode (FNode): The FNode representing the condition to be added as a constraint.
            name (str): The name of the condition for identification within the model.
        """

        # TODO: support open intervals and delay
        if time_interval.lower.delay != 0:
            raise NotImplementedError("Timing delay not supported.")
        if time_interval.is_left_open():
            raise NotImplementedError("Open intervals not supported.")
        start = self.model_vars[time_interval.lower.timepoint]

        if time_interval.upper.delay != 0:
            raise NotImplementedError("Timing delay not supported.")
        if time_interval.is_right_open():
            raise NotImplementedError("Open intervals not supported.")
        end = self.model_vars[time_interval.upper.timepoint]

        # if there are no fluents in the condition, a constraint should be added
        if not self._fnode_contains_fluents(fnode):
            constraint_var = self.add_constraint(fnode)
            self.model.add(start <= end).only_enforce_if(constraint_var)
            self.model.add(start > end).only_enforce_if(constraint_var.negated())
            return

        # for each timepoint, add a constraint defined on the fluents at that timepoint
        # and enforce:
        #   (start <= timepoint <= end) => constraint
        for i in range(len(self.timepoints)):
            for fluent in problem.fluents:
                self.model_vars[fluent] = self.resources[fluent.name][i + 1]

            constraint_var = self.add_constraint(fnode, cache_enabled=False)

            tp_GE_start_key = f"{self.timepoints[i].name} >= {start.name}"
            if tp_GE_start_key not in self._variables_cache:
                tp_GE_start = self.new_bool_var()
                self.model.add(self.timepoints[i] >= start).only_enforce_if(tp_GE_start)
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
                self.model.add_bool_or(
                    [tp_GE_start.negated(), tp_LE_end.negated(), constraint_var]
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
                self.model.add_bool_or(
                    [tp_GE_start.negated(), tp_LE_end.negated(), constraint_var]
                ).only_enforce_if(next_timepoint_is_different)

            # self.timepoints[i+1] == self.timepoints[i]

    def add_conditions(self, problem: SchedulingProblem):
        """
        Adds the conditions of the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                conditions to be added to the model.
        """

        for time_interval, fnode in problem.base_conditions:
            self.add_condition(time_interval, fnode, problem)

        for act in problem.activities:
            for time_interval in act.conditions:
                for fnode in act.conditions[time_interval]:
                    self.add_condition(time_interval, fnode, problem)
