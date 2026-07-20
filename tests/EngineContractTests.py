# Copyright (C) 2025 PSO Unit, Fondazione Bruno Kessler
# This file is part of CPSE.
#
# CPSE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CPSE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

from abc import abstractmethod

import pytest
from unified_planning.engines import PlanGenerationResult, PlanGenerationResultStatus
from unified_planning.environment import get_environment
from unified_planning.model import (
    ClosedTimeInterval,
    GlobalStartTiming,
    LeftOpenTimeInterval,
    MinimizeMakespan,
    MinimizeSequentialPlanLength,
    OpenTimeInterval,
    RightOpenTimeInterval,
    TimeInterval,
    Timepoint,
    TimePointInterval,
    TimepointKind,
    Timing,
)
from unified_planning.model.scheduling import Activity, SchedulingProblem
from unified_planning.plans import Schedule
from unified_planning.shortcuts import (
    LE,
    And,
    BoolType,
    Equals,
    Iff,
    Implies,
    IntType,
    Minus,
    Not,
    OneshotPlanner,
    Or,
    Plus,
    Times,
    UserType,
)


def are_activities_overlapped(plan: Schedule, activities: list[Activity]) -> bool:
    def start_time(activity):
        return plan.get(activity.start).constant_value()

    sorted_activities = sorted(activities, key=start_time)
    for i in range(1, len(sorted_activities)):
        end_prev = plan.get(sorted_activities[i - 1].end).constant_value()
        start = plan.get(sorted_activities[i].start).constant_value()
        if end_prev > start:
            return True
    return False


class EngineContractTests:
    @abstractmethod
    def engine_name(self) -> str:
        pass

    @abstractmethod
    def engine_class(self):
        pass

    def problem_solved_satisficing_or_optimally(
        self, problem: SchedulingProblem
    ) -> PlanGenerationResult:
        with OneshotPlanner(name=self.engine_name()) as planner:
            res = planner.solve(problem)
            print(res)
            print(res.log_messages)
            assert res.status in [
                PlanGenerationResultStatus.SOLVED_SATISFICING,
                PlanGenerationResultStatus.SOLVED_OPTIMALLY,
            ]
            assert res.plan is not None
            assert isinstance(res.plan, Schedule)
            return res
        raise Exception(f"{self.engine_name()} engine cannot be loaded")

    def problem_unsolvable(self, problem: SchedulingProblem) -> PlanGenerationResult:
        with OneshotPlanner(name=self.engine_name()) as planner:
            res = planner.solve(problem)
            print(res)
            print(res.log_messages)
            assert res.status in [
                PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
                PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
            ]
            assert res.plan is None
            return res
        raise Exception(f"{self.engine_name()} engine cannot be loaded")

    def problem_unsupported(self, problem: SchedulingProblem) -> PlanGenerationResult:
        with OneshotPlanner(name=self.engine_name()) as planner:
            res = planner.solve(problem)
            print(res)
            print(res.log_messages)
            assert res.status == PlanGenerationResultStatus.UNSUPPORTED_PROBLEM
            assert res.plan is None
            return res
        raise Exception(f"{self.engine_name()} engine cannot be loaded")

    def test_new_engine(self):
        planner = self.engine_class()(lower_bound=1, upper_bound=2)
        assert planner.lower_bound == 1
        assert planner.upper_bound == 2

        with OneshotPlanner(
            name=self.engine_name(), params={"lower_bound": 3, "upper_bound": 100}
        ) as planner:
            assert isinstance(planner, self.engine_class())
            assert planner.lower_bound == 3
            assert planner.upper_bound == 100

    def test_activity_uses_resource(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=1)
        activity = problem.add_activity("activity", duration=1)
        activity.uses(resource, 1)
        self.problem_solved_satisficing_or_optimally(problem)

    def test_activity_uses_more_resources(self, problem: SchedulingProblem):
        resource1 = problem.add_resource("resource1", capacity=1)
        resource2 = problem.add_resource("resource2", capacity=2)
        resource3 = problem.add_resource("resource3", capacity=3)

        activity = problem.add_activity("activity", duration=1)
        activity.uses(resource1, 1)
        activity.uses(resource2, 2)
        activity.uses(resource3, 3)

        self.problem_solved_satisficing_or_optimally(problem)

    def test_activities_use_same_resource(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=1)

        activity1 = problem.add_activity("activity1", duration=1)
        activity1.uses(resource, 1)

        activity2 = problem.add_activity("activity2", duration=1)
        activity2.uses(resource, 1)

        activity3 = problem.add_activity("activity3", duration=1)
        activity3.uses(resource, 1)

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert not are_activities_overlapped(res.plan, res.plan.activities)

    def test_set_activity_duration_bounds(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=1)
        activity.set_duration_bounds(10, 14)
        activity.add_constraint(Equals(activity.start, 0))
        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity.start).constant_value() == 0
        assert 10 <= res.plan.get(activity.end).constant_value() <= 14

    def test_activity_duration_as_parameter_exp1(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )
        problem.add_constraint(Equals(int_var, 5))
        activity.set_fixed_duration(Times(int_var, 2))

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert (
            res.plan.get(activity.end).constant_value()
            - res.plan.get(activity.start).constant_value()
        ) == 10

    def test_activity_duration_as_parameter_exp2(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )
        problem.add_constraint(Equals(activity.start, 0))
        problem.add_constraint(Equals(int_var, 5))
        activity.set_duration_bounds(Times(int_var, 2), Times(int_var, 3))

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity.start).constant_value() == 0
        assert 10 <= res.plan.get(activity.end).constant_value() <= 15

    def test_activity_deadline_and_release_date(self, problem: SchedulingProblem):
        activity1 = problem.add_activity("activity1", duration=5)
        activity2 = problem.add_activity("activity2", duration=5)
        activity1.add_deadline(Plus(activity2.end, 3))
        activity1.add_release_date(Minus(activity2.start, 3))

        res = self.problem_solved_satisficing_or_optimally(problem)
        activity1_start = res.plan.get(activity1.start).constant_value()
        activity1_end = res.plan.get(activity1.end).constant_value()
        activity2_start = res.plan.get(activity2.start).constant_value()
        activity2_end = res.plan.get(activity2.end).constant_value()
        assert (activity2_start - 3) <= activity1_start
        assert activity1_end <= (activity2_end + 3)

    @pytest.mark.skip(reason="ortools solver takes too long")
    def test_set_fluent_initial_value(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=2)
        problem.set_initial_value(resource, 1)

        activity = problem.add_activity("activity", duration=1)
        activity.uses(resource, 2)

        self.problem_unsolvable(problem)

    def test_problem_variables(self, problem: SchedulingProblem):
        problem.add_activity("activity", duration=1)

        bool_var = problem.add_variable(
            "bool_var", get_environment().type_manager.BoolType()
        )
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert isinstance(res.plan.get(bool_var).constant_value(), bool)
        assert res.plan.get(int_var).constant_value() >= 0

    def test_activity_parameters(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=1)

        bool_var = activity.add_parameter(
            "bool_var", get_environment().type_manager.BoolType()
        )
        int_var = activity.add_parameter(
            "int_var", get_environment().type_manager.IntType()
        )

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert isinstance(res.plan.get(bool_var).constant_value(), bool)
        assert res.plan.get(int_var).constant_value() >= 0

    def test_bounded_int_parameters(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=1)

        int_var1 = problem.add_variable(
            "int_var1",
            get_environment().type_manager.IntType(lower_bound=10, upper_bound=11),
        )
        int_var2 = activity.add_parameter(
            "int_var2",
            get_environment().type_manager.IntType(lower_bound=20, upper_bound=21),
        )

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert 10 <= res.plan.get(int_var1).constant_value() <= 11
        assert 20 <= res.plan.get(int_var2).constant_value() <= 21

    def test_problem_constraints(self, problem: SchedulingProblem):
        activity1 = problem.add_activity("activity1", duration=1)
        activity2 = problem.add_activity("activity2", duration=1)
        activity1_first = problem.add_variable(
            "activity1_first", get_environment().type_manager.BoolType()
        )

        problem.add_constraint(
            Or(
                Equals(activity1.end, activity2.start),
                Equals(activity1.start, activity2.end),
            )
        )
        problem.add_constraint(
            And(
                Implies(Equals(activity1.end, activity2.start), activity1_first),
                Implies(Equals(activity1.start, activity2.end), Not(activity1_first)),
            )
        )
        problem.add_constraint(
            Iff(Equals(activity1.end, activity2.start), activity1_first)
        )
        problem.add_constraint(Equals(activity1.end, activity2.start))

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity1_first).constant_value()
        assert (
            res.plan.get(activity1.end).constant_value()
            == res.plan.get(activity2.start).constant_value()
        )

    def test_activity_constraints(self, problem: SchedulingProblem):
        activity1 = problem.add_activity("activity1", duration=1)
        activity2 = problem.add_activity("activity2", duration=1)
        activity1_first = problem.add_variable(
            "activity1_first", get_environment().type_manager.BoolType()
        )

        activity1.add_constraint(
            Or(
                Equals(activity1.end, activity2.start),
                Equals(activity1.start, activity2.end),
            )
        )
        activity2.add_constraint(
            And(
                Implies(Equals(activity1.end, activity2.start), activity1_first),
                Implies(Equals(activity1.start, activity2.end), Not(activity1_first)),
            )
        )
        activity1.add_constraint(
            Iff(Equals(activity1.end, activity2.start), activity1_first)
        )
        activity2.add_constraint(Equals(activity1.end, activity2.start))

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity1_first).constant_value()
        assert (
            res.plan.get(activity1.end).constant_value()
            == res.plan.get(activity2.start).constant_value()
        )

    def test_problem_effects(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=1)
        activity = problem.add_activity("activity", duration=20)

        problem.add_decrease_effect(activity.start, resource, 1)
        problem.add_increase_effect(10, resource, 1)

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity.start).constant_value() <= 10

    def test_activity_effects(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=4)
        problem.set_initial_value(resource, 0)

        activity1 = problem.add_activity("activity1", duration=4)
        activity2 = problem.add_activity("activity2", duration=4)
        activity3 = problem.add_activity("activity3", duration=4)

        problem.add_increase_effect(10, resource, 2)
        activity1.add_decrease_effect(activity1.start, resource, 2)

        problem.add_increase_effect(20, resource, 3)
        activity2.add_decrease_effect(activity2.start, resource, 3)

        problem.add_increase_effect(30, resource, 4)
        activity3.add_decrease_effect(activity3.start, resource, 4)

        self.problem_solved_satisficing_or_optimally(problem)

    def test_multiple_effects_at_the_same_timepoint(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=2)
        activity = problem.add_activity("activity", duration=20)

        problem.add_decrease_effect(activity.start, resource, 2)
        problem.add_increase_effect(activity.start, resource, 1)

        self.problem_solved_satisficing_or_optimally(problem)

    def test_conditional_effects(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=2)
        problem.set_initial_value(resource, 0)

        activity1 = problem.add_activity("activity1", duration=20)
        activity2 = problem.add_activity("activity2", duration=20)

        problem.add_increase_effect(
            activity1.start, resource, 1, condition=LE(10, activity1.start)
        )
        problem.add_decrease_effect(activity2.start, resource, 1)

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity1.start).constant_value() >= 10

    def test_problem_conditions(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=10)
        bool_var = problem.add_variable(
            "bool_var", get_environment().type_manager.BoolType()
        )
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )

        problem.add_condition(
            ClosedTimeInterval(Timing(0, activity.start), Timing(0, activity.end)),
            LE(5, int_var),
        )
        problem.add_condition(
            TimeInterval(
                lower=Timing(
                    delay=0,
                    timepoint=Timepoint(TimepointKind.GLOBAL_START, container=None),
                ),
                upper=Timing(
                    delay=0,
                    timepoint=Timepoint(TimepointKind.GLOBAL_END, container=None),
                ),
                is_left_open=True,
                is_right_open=True,
            ),
            LE(int_var, 10),
        )
        problem.add_condition(
            TimeInterval(
                lower=Timing(1, activity.start),
                upper=Timing(2, activity.end),
                is_left_open=True,
                is_right_open=True,
            ),
            Implies(LE(int_var, 15), bool_var),
        )
        problem.add_condition(
            TimeInterval(
                lower=GlobalStartTiming(10),
                upper=GlobalStartTiming(15),
                is_left_open=True,
                is_right_open=False,
            ),
            bool_var,
        )
        res = self.problem_solved_satisficing_or_optimally(problem)
        assert 5 <= res.plan.get(int_var).constant_value() <= 10
        assert res.plan.get(bool_var).constant_value()

    def test_activity_conditions(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=10)
        bool_var = problem.add_variable(
            "bool_var", get_environment().type_manager.BoolType()
        )
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )

        activity.add_condition(
            TimeInterval(
                lower=Timing(delay=1, timepoint=activity.start),
                upper=Timing(delay=0, timepoint=activity.end),
                is_left_open=True,
                is_right_open=False,
            ),
            LE(int_var, 10),
        )
        activity.add_condition(
            TimePointInterval(Timing(3, activity.start)),
            LE(5, int_var),
        )
        activity.add_condition(
            OpenTimeInterval(Timing(3, activity.start), Timing(-2, activity.end)),
            LE(5, int_var),
        )
        activity.add_condition(
            ClosedTimeInterval(Timing(3, activity.start), Timing(-2, activity.end)),
            LE(5, int_var),
        )
        activity.add_condition(
            LeftOpenTimeInterval(Timing(3, activity.start), Timing(-2, activity.end)),
            LE(5, int_var),
        )
        activity.add_condition(
            RightOpenTimeInterval(Timing(3, activity.start), Timing(-2, activity.end)),
            LE(5, int_var),
        )

        problem.add_constraint(Implies(And(LE(1, int_var), LE(int_var, 20)), bool_var))
        res = self.problem_solved_satisficing_or_optimally(problem)
        assert 5 <= res.plan.get(int_var).constant_value() <= 10
        assert res.plan.get(bool_var).constant_value()

    def test_minimize_makespan(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=1)
        duration = 10
        num_activities = 5
        activities = []
        for i in range(num_activities):
            activity = problem.add_activity(f"activity{i}", duration=duration)
            activity.uses(resource, 1)
            activities.append(activity)

        problem.add_quality_metric(MinimizeMakespan())

        res = self.problem_solved_satisficing_or_optimally(problem)
        for activity in activities:
            assert res.plan.get(activity.end).constant_value() <= (
                num_activities * duration
            )

    def test_int_fluent_with_constant_values(self, problem: SchedulingProblem):
        fluent = problem.add_fluent(
            "fluent",
            get_environment().type_manager.IntType(lower_bound=0),
            default_initial_value=0,
        )

        activity = problem.add_activity("activity", duration=5)
        activity.add_increase_effect(activity.start + 1, fluent, 1)
        activity.add_decrease_effect(activity.end, fluent, 1)

        self.problem_solved_satisficing_or_optimally(problem)

    @pytest.mark.skip(reason="ortools solver takes too long")
    def test_fluent_set_initial_value(self, problem: SchedulingProblem):
        fluent = problem.add_fluent(
            "fluent",
            get_environment().type_manager.IntType(lower_bound=0, upper_bound=1),
        )
        problem.set_initial_value(fluent, 1)

        activity = problem.add_activity("activity", duration=5)
        activity.add_increase_effect(activity.start + 1, fluent, 1)
        activity.add_decrease_effect(activity.end, fluent, 1)

        self.problem_unsolvable(problem)

    def test_bool_fluent_with_constant_value(self, problem: SchedulingProblem):
        fluent = problem.add_fluent("fluent", default_initial_value=True)
        problem.set_initial_value(fluent, False)
        problem.add_activity("activity", duration=5)
        self.problem_solved_satisficing_or_optimally(problem)

    def test_fluent_with_args(self, problem: SchedulingProblem):
        user_type_parent = UserType("user_type_parent")
        user_type = UserType("user_type", user_type_parent)
        fluent = problem.add_fluent(
            "fluent",
            IntType(lower_bound=0, upper_bound=10),
            obj=user_type_parent,
            default_initial_value=1,
        )
        o1 = problem.add_object("o1", user_type_parent)
        o2 = problem.add_object("o2", user_type)

        activity = problem.add_activity("activity", 2)
        activity.add_increase_effect(activity.end, fluent(o1), 1)
        activity.add_decrease_effect(activity.end, fluent(o2), 1)

        self.problem_solved_satisficing_or_optimally(problem)

    def test_constraint_with_constant_bool(self, problem: SchedulingProblem):
        problem.add_activity("activity", 1)
        bool_var = problem.add_variable("bool_var", BoolType())

        problem.add_constraint(True)
        problem.add_constraint(Not(False))
        problem.add_constraint(problem.environment.expression_manager.TRUE())
        problem.add_constraint(Not(problem.environment.expression_manager.FALSE()))
        problem.add_constraint(Not(Not(True)))
        problem.add_constraint(And(True))
        problem.add_constraint(And(bool_var, True))
        problem.add_constraint(And(Or(bool_var, False), True))
        problem.add_constraint(Or(And(bool_var, False), True))
        problem.add_constraint(Implies(Or(bool_var, False), True))

        self.problem_solved_satisficing_or_optimally(problem)

    def test_simple_problem(self, problem: SchedulingProblem):
        machine = problem.add_resource("machine", capacity=1)

        # Define activities with specific durations
        activity1 = problem.add_activity("activity1", duration=5)
        activity2 = problem.add_activity("activity2", duration=10)
        activity3 = problem.add_activity("activity3", duration=4)

        activity2.add_constraint(Equals(activity2.start, 5))
        activity2.add_constraint(Equals(activity2.end, 15))

        # Each activity requires the machine resource
        activity2.uses(machine, 1)
        activity3.uses(machine, 1)

        # activity1.uses(machine, 1)
        problem.add_decrease_effect(activity1.start, machine, 1)
        problem.add_increase_effect(activity1.end, machine, 1)

        problem.add_constraint(Equals(activity2.end, activity3.start))

        activity1_before_activity2 = problem.add_variable(
            "activity1_before_activity2", get_environment().type_manager.BoolType()
        )
        activity2_before_activity3 = problem.add_variable(
            "activity2_before_activity3", get_environment().type_manager.BoolType()
        )
        problem.add_constraint(
            Iff(LE(activity1.end, activity2.start), activity1_before_activity2)
        )
        problem.add_constraint(
            Iff(LE(activity2.end, activity3.start), activity2_before_activity3)
        )

        # minimizing makespan, activity1 should be the first activity
        problem.add_quality_metric(MinimizeMakespan())

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity1_before_activity2).constant_value()
        assert res.plan.get(activity2_before_activity3).constant_value()
        assert res.plan.get(activity3.end).constant_value() == 19
        assert not are_activities_overlapped(
            res.plan, [activity1, activity2, activity3]
        )

    def test_not_supported_parameter_type(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)
        problem.add_variable("param1", get_environment().type_manager.RealType())
        activity.add_parameter("param2", get_environment().type_manager.RealType())
        assert not self.engine_class().supports(problem.kind)

    def test_not_supported_quality_metrics(self, problem: SchedulingProblem):
        problem.add_activity("activity", duration=5)
        problem.add_quality_metric(MinimizeSequentialPlanLength())
        assert not self.engine_class().supports(problem.kind)
