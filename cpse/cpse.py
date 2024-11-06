#!/usr/bin/env python3
"""Unified Planning Integration for OR-Tools CP-SAT Model"""
from typing import IO, Callable, Optional, List, Union, Any, Dict
from abc import abstractmethod
import collections
import operator
import warnings

import unified_planning as up
from unified_planning.engines import (
    Credits,
    PlanGenerationResult,
    PlanGenerationResultStatus,
)
from unified_planning.engines.mixins import OptimalityGuarantee
from unified_planning.model import OperatorKind, Parameter, FNode, timing, ProblemKind
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
    """CPSE Base Engine abstract class."""

    def __init__(self, **kwargs):
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)

        self.lower_bound: int = kwargs.get("lower_bound", 0)
        self.upper_bound: int = kwargs.get("upper_bound", cp_model.INT32_MAX)
        self.model = cp_model.CpModel()

        self.activities: Dict[str, activity_type] = {}
        self.model_vars: Dict[Union[timing.Timepoint, Parameter], cp_model.IntVar] = {}
        self.fluent_capacity: Dict[str, int] = {}
        self.fluent_initial_value: Dict[str, int] = {}

        self.global_start = timing.GlobalStartTiming(delay=0)
        self.global_end = timing.GlobalEndTiming()

        self.bool_var_counter: int = -1
        self.int_var_counter: int = -1

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
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_parameters("BOOL_ACTION_PARAMETERS")
        supported_kind.set_parameters("UNBOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_quality_metrics("MAKESPAN")
        return supported_kind

    @staticmethod
    def supports(problem_kind: ProblemKind) -> bool:
        return problem_kind <= CPSE.supported_kind()

    def new_bool_var(self):
        """Add a new anonymous boolean variable to the model."""
        self.bool_var_counter += 1
        return self.model.new_bool_var(f"bool{self.bool_var_counter}")

    def new_int_var(self):
        """Add a new anonymous integer variable to the model."""
        self.int_var_counter += 1
        return self.model.new_int_var(
            self.lower_bound, self.upper_bound, f"int{self.int_var_counter}"
        )

    def _convert_timing_to_linear_expr(self, t: timing.Timing) -> cp_model.LinearExprT:
        """Convert a `timing.Timing` variable to a `cp_model.LinearExprT`."""
        assert t.timepoint in self.model_vars
        assert isinstance(t.delay, int)
        return cp_model.LinearExpr.sum([self.model_vars[t.timepoint], t.delay])

    def add_parameters(self, parameters: List[Parameter]):
        """Add the parameters to the model as boolean or integer variables."""

        for param in parameters:
            if param.type.is_bool_type():
                var = self.model.new_bool_var(param.name)
            elif param.type.is_int_type():
                var = self.model.new_int_var(
                    self.lower_bound, self.upper_bound, param.name
                )
            else:
                raise NotImplementedError(f"Parameter type {param.type} not supported.")

            assert param not in self.model_vars
            self.model_vars[param] = var

    def add_activity(self, activity: Activity):
        """Add an activity to the model. Each activity is modeled using an interval."""

        # assume FixedDuration or ClosedDurationInterval
        assert (
            activity.duration.lower.is_int_constant()
            and activity.duration.upper.is_int_constant()
            and not activity.duration.is_left_open()
            and not activity.duration.is_right_open()
        )

        lower = activity.duration.lower.constant_value()
        upper = activity.duration.upper.constant_value()

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

    def add_constraint(
        self, fnode: FNode
    ) -> Union[cp_model.IntVar, cp_model.LinearExprT, bool, int, Any]:
        """Add the constraint represented by the fnode to the model."""

        # TODO: reuse bool_var for the same constraint
        # TODO: transform to normal form?
        # TODO: use cache

        stack = [(fnode, False)]
        results = []

        while len(stack) > 0:
            fnode, processed = stack.pop()

            if fnode.is_parameter_exp():
                results.append(self.model_vars[fnode.parameter()])

            elif fnode.is_timing_exp():
                results.append(self._convert_timing_to_linear_expr(fnode.timing()))

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

            else:
                raise NotImplementedError(f"Node type {fnode.node_type} not supported.")

        assert len(results) == 1
        return results[0]

    def add_constraints(self, problem: SchedulingProblem):
        """Add the problem constraints to the model."""

        # TODO: avoid bool_var for the root node

        for fnode in problem.base_constraints:
            bool_var = self.add_constraint(fnode)
            self.model.add_bool_and([bool_var])

        for act in problem.activities:
            for fnode in act.constraints:
                bool_var = self.add_constraint(fnode)
                self.model.add_bool_and([bool_var])

    @abstractmethod
    def add_effects(self, problem: SchedulingProblem):
        pass

    @abstractmethod
    def add_conditions(self, problem: SchedulingProblem):
        pass

    def add_quality_metrics(self, problem: SchedulingProblem, makespan_var):
        """Add the quality metrics to the model."""

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
        assert isinstance(
            problem, SchedulingProblem
        ), f"problem of type {type(problem)} not supported"
        if heuristic is not None:
            warnings.warn("CPSE does not support custom heuristics", UserWarning)

        self.model.name = problem.name

        # map each fluent to its maximum capacity and initial value
        for f in problem.fluents:
            self.fluent_capacity[f.name] = problem.fluents_defaults[f].constant_value()
            self.fluent_initial_value[f.name] = problem.fluents_defaults[
                f
            ].constant_value()

        # override initial values when an explicit one is defined
        for f in problem.explicit_initial_values:
            self.fluent_initial_value[f.fluent().name] = (
                problem.explicit_initial_values[f].constant_value()
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
        self.model_vars[self.global_start.timepoint] = 0
        self.model_vars[self.global_end.timepoint] = makespan_var

        # add the effects to the model
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


class CPSE(CPSEBaseEngine):
    """Implementation of the CPSE Engine."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "CPSE"

    def add_effects(self, problem: SchedulingProblem):
        """Add a reservoir constraint for each fluent. The value of the fluent
        is constrained to remain between [0, C] where C is its maximum capacity."""

        # TODO: model conditions on effects

        activities_effects = [
            (timing, eff)
            for act in problem.activities
            for timing, effects in act.effects.items()
            for eff in effects
        ]

        # map each fluent to its effects
        fluent_effects: Dict[str, Dict["timing.Timing", int]] = {}
        for timing, eff in activities_effects + problem.base_effects:
            fluent_name = eff.fluent.fluent().name
            value = eff.value.constant_value()
            if eff.is_increase():
                pass
            elif eff.is_decrease():
                value = -value
            else:
                raise NotImplementedError(f"Effect kind {eff.kind} not supported.")

            if fluent_name not in fluent_effects:
                fluent_effects[fluent_name] = {}

            if timing not in fluent_effects[fluent_name]:
                fluent_effects[fluent_name][timing] = value
            else:
                # sum values of different effects that occur at the same timepoint
                fluent_effects[fluent_name][timing] += value

        # add a reservoir constraint for each fluent: the value of the fluent
        # should remain between [0, C] where C is its maximum capacity
        for fluent_name in fluent_effects:
            times = [0]
            values = [self.fluent_initial_value[fluent_name]]
            for timing in fluent_effects[fluent_name]:
                value = fluent_effects[fluent_name][timing]
                if value != 0:  # when value is 0, the effect is ignored
                    times.append(self._convert_timing_to_linear_expr(timing))
                    values.append(value)

            self.model.add_reservoir_constraint(
                times, values, 0, self.fluent_capacity[fluent_name]
            )

    def add_condition(
        self, time_interval: timing.TimeInterval, fnode: FNode, name: str
    ):
        """Add a condition to the model. Add a constraint and force it is satisfied
        in the time interval using the `add_cumulative` method."""

        bool_var = self.add_constraint(fnode)

        start = self.model_vars[time_interval.lower.timepoint]
        if time_interval.lower.delay != 0:
            assert isinstance(time_interval.lower.delay, int)
            start += time_interval.lower.delay
        if time_interval.is_left_open():
            start += 1

        # end + 1 because `add_cumulative` enforce the constraint for all t
        # st. start <= t < end
        end = self.model_vars[time_interval.upper.timepoint] + 1
        if time_interval.upper.delay != 0:
            assert isinstance(time_interval.upper.delay, int)
            end += time_interval.upper.delay
        if time_interval.is_right_open():
            end -= 1

        duration = self.model.new_int_var(
            self.lower_bound,
            self.upper_bound,
            f"{name}_duration",
        )

        # TODO: reuse interval if present
        interval_var = self.model.new_interval_var(start, duration, end, name)
        self.model.add_cumulative([interval_var], [bool_var.negated()], 0)

    def add_conditions(self, problem: SchedulingProblem):
        """Add the conditions defined in the problem and the activities to the model."""

        # TODO: conditions cannot be defined on fluents

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timepoints = None
        self.assignment_matrix = None
        self.resources = None

    @property
    def name(self) -> str:
        return "CPSETimepoints"

    def timepoints_setup(self, problem: SchedulingProblem):
        self.timepoints = [
            self.model.new_int_var(self.lower_bound, self.upper_bound, f"timepoint{i}")
            for i in range(len(problem.activities) * 2)
        ]
        for i in range(len(self.timepoints) - 1):
            self.model.add(self.timepoints[i] <= self.timepoints[i + 1])
            print(f"{self.timepoints[i]} <= {self.timepoints[i + 1]}")

        activity_timepoints = []
        for activity_name in sorted(self.activities.keys()):
            activity = self.activities[activity_name]
            activity_timepoints.append((activity.up_activity.start, activity.start))
            activity_timepoints.append((activity.up_activity.end, activity.end))
        self.assignment_matrix = {}
        # position_vars = []
        # all_comparison_vars = []
        for i in range(len(activity_timepoints)):
            up_tp_i, activity_timepoint_i = activity_timepoints[i]
            # comparison_vars = []
            # for j in range(len(activity_timepoints)):
            #     if i == j:
            #         continue

            #     up_tp_j, activity_timepoint_j = activity_timepoints[j]
            #     comparison_var = self.new_bool_var()
            #     comparison_vars.append(comparison_var)
            #     all_comparison_vars.append(comparison_var)
            #     # self.model.add(
            #     #     (activity_timepoint_i > activity_timepoint_j) == comparison_var
            #     # )
            #     self.model.add(
            #         activity_timepoint_i > activity_timepoint_j
            #     ).only_enforce_if(comparison_var)
            #     print(
            #         f"{comparison_var} => ({activity_timepoint_i} > {activity_timepoint_j})"
            #     )

            # position_var = self.model.new_int_var(0, len(activity_timepoints)-1, f"activity_tp{i}_position")
            # position_vars.append(position_var)
            # self.model.add(position_var == cp_model.LinearExpr.sum(comparison_vars))

            self.assignment_matrix[up_tp_i] = []
            for idx, tp in enumerate(self.timepoints):
                # if tp is idx-th <=> (tp_i == activity_timepoint_i)
                # self.model.add((sum(comparison_vars) == idx) == (tp == activity_timepoint_i))
                # self.model.add((sum(comparison_vars) == idx) == bool_var)
                # self.model.add((tp == activity_timepoint_i) == bool_var)

                bool_var = self.model.new_bool_var(f"{activity_timepoint_i}@{tp}")
                self.assignment_matrix[up_tp_i].append(bool_var)
                # self.model.add(
                #     cp_model.LinearExpr.sum(comparison_vars) == idx
                # ).only_enforce_if(bool_var)
                self.model.add(tp == activity_timepoint_i).only_enforce_if(bool_var)
                # print(f"{bool_var} => ({sum(comparison_vars)} == {idx})")
                print(f"{bool_var} => ({tp} == {activity_timepoint_i})")

            # each activity timepoint is assigned to exactly one timepoint
            print(f"exactly_one({self.assignment_matrix[up_tp_i]})")
            self.model.add_exactly_one(self.assignment_matrix[up_tp_i])

        # each timepoint is assigned to exactly one activity timepoint
        for i in range(len(self.timepoints)):
            self.model.add_exactly_one(
                bool_vars[i] for bool_vars in self.assignment_matrix.values()
            )
            print(
                f"exactly_one({[bool_vars[i] for bool_vars in self.assignment_matrix.values()]})"
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

    def _filter_fluent_effects(self, problem, fluent):
        activity_effects = [
            (timing, eff)
            for act in problem.activities
            for timing, effects in act.effects.items()
            for eff in effects
            if eff.fluent.fluent().name == fluent.name
        ]

        problem_effects = list(
            filter(
                lambda t_eff: t_eff[1].fluent.fluent().name == fluent.name,
                problem.base_effects,
            )
        )

        return activity_effects + problem_effects

    def add_assign_effect_constraints(self, problem):
        visited_timepoints = set()
        for fluent in problem.fluents:
            fluent_effects = self._filter_fluent_effects(problem, fluent)
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
            print(f"all_different{activity_timepoints}")

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

                    print(
                        f"{self.model_vars[t1.timepoint]} != {self.model_vars[t2.timepoint]}"
                    )
                    self.model.add(
                        self.model_vars[t1.timepoint] != self.model_vars[t2.timepoint]
                    )

    def all_tpi_with_effects_on_fluent(self, tpi, fluent_name, problem):
        for act in problem.activities:
            for timing, effects in act.effects.items():
                for eff in effects:
                    if eff.fluent.fluent().name == fluent_name:
                        yield self.assignment_matrix[timing.timepoint][tpi]

        for timing, eff in problem.base_effects:
            if eff.fluent.fluent().name == fluent_name:
                yield self.assignment_matrix[timing.timepoint][tpi]

    def add_effect(self, timing, effect):
        fluent_name = effect.fluent.fluent().name
        value = effect.value.constant_value()
        assert value >= 0
        assert timing.delay == 0

        # TODO: support global timings and delays

        if effect.is_decrease():
            value = -value

        for i, bool_var in enumerate(self.assignment_matrix[timing.timepoint]):
            if effect.is_assignment():
                self.model.add(
                    self.resources[fluent_name][i + 1] == value
                ).only_enforce_if(bool_var)
                print(
                    f"{bool_var} => ({self.resources[fluent_name][i + 1]} == {value})"
                )

            else:
                self.model.add(
                    self.resources[fluent_name][i + 1]
                    == (self.resources[fluent_name][i] + value)
                ).only_enforce_if(bool_var)
                print(
                    f"{bool_var} => ({self.resources[fluent_name][i + 1]} == {(self.resources[fluent_name][i] + value)})"
                )

    def add_effects(self, problem: SchedulingProblem):
        self.timepoints_setup(problem)

        for act in problem.activities:
            for timing, effects in act.effects.items():
                from unified_planning.model.timing import Timepoint

                assert isinstance(timing.timepoint, Timepoint)
                for eff in effects:
                    self.add_effect(timing, eff)

        for i, (timing, eff) in enumerate(problem.base_effects):
            self.add_effect(timing, eff)

        # for each resource
        #   for each timepoint
        #       if no activities have effects on that resource at that timepoint
        #           resource@tp(i) == resource@tp(i-1)
        for fluent_name in self.resources:
            for i, tp in enumerate(self.timepoints):
                activity_tps = list(
                    self.all_tpi_with_effects_on_fluent(i, fluent_name, problem)
                )
                bool_var = self.new_bool_var()
                self.model.add(
                    self.resources[fluent_name][i + 1] == self.resources[fluent_name][i]
                ).only_enforce_if(bool_var)
                self.model.add_bool_or(activity_tps + [bool_var])
                print(
                    f"or({activity_tps}) or ({self.resources[fluent_name][i + 1]} == {self.resources[fluent_name][i]})"
                )

        self.add_assign_effect_constraints(problem)

    def add_condition(
        self, time_interval: timing.TimeInterval, fnode: FNode, name: str
    ):
        """Add a condition to the model. Add a constraint and force it is satisfied
        in the time interval using the `add_cumulative` method."""

        # TODO: enforce condition using timepoints

        bool_var = self.add_constraint(fnode)

        start = self.model_vars[time_interval.lower.timepoint]
        if time_interval.lower.delay != 0:
            assert isinstance(time_interval.lower.delay, int)
            start += time_interval.lower.delay
        if time_interval.is_left_open():
            start += 1

        # end + 1 because `add_cumulative` enforce the constraint for all t
        # st. start <= t < end
        end = self.model_vars[time_interval.upper.timepoint] + 1
        if time_interval.upper.delay != 0:
            assert isinstance(time_interval.upper.delay, int)
            end += time_interval.upper.delay
        if time_interval.is_right_open():
            end -= 1

        duration = self.model.new_int_var(
            self.lower_bound,
            self.upper_bound,
            f"{name}_duration",
        )

        # TODO: reuse interval if present
        interval_var = self.model.new_interval_var(start, duration, end, name)
        self.model.add_cumulative([interval_var], [bool_var.negated()], 0)

    def add_conditions(self, problem: SchedulingProblem):
        """Add the conditions defined in the problem and the activities to the model."""

        # TODO: conditions cannot be defined on fluents

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
