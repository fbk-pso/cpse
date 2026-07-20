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

from unified_planning.environment import get_environment
from unified_planning.model import (
    ClosedTimeInterval,
    MinimizeMakespan,
    TimePointInterval,
    Timing,
)
from unified_planning.model.scheduling import SchedulingProblem
from unified_planning.shortcuts import (
    GE,
    GT,
    LE,
    LT,
    And,
    BoolType,
    Equals,
    Iff,
    IntType,
    Not,
    Plus,
    Times,
    UserType,
)

from cpse import CPSETimepoints

from .EngineContractTests import EngineContractTests


class TestCPSETimepoints(EngineContractTests):
    def engine_name(self):
        return "cpse-timepoints"

    def engine_class(self):
        return CPSETimepoints

    def test_activity_duration_as_fluent_exp1(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)
        fluent = problem.add_fluent(
            "fluent",
            get_environment().type_manager.IntType(),
            default_initial_value=5,
        )
        activity.set_fixed_duration(Times(fluent, 2))

        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert (
            schedule.get(activity.end).constant_value()
            - schedule.get(activity.start).constant_value()
        ) == 10

    def test_activity_duration_as_fluent_exp2(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)
        fluent = problem.add_fluent(
            "fluent",
            get_environment().type_manager.IntType(),
            default_initial_value=5,
        )
        problem.add_constraint(Equals(activity.start, 0))
        activity.set_duration_bounds(Times(fluent, 2), Times(fluent, 3))

        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert schedule.get(activity.start).constant_value() == 0
        assert 10 <= schedule.get(activity.end).constant_value() <= 15

    def test_activity_duration_with_parametric_fluents(
        self, problem: SchedulingProblem
    ):

        user_type = UserType("user_type")
        param = problem.add_variable("param", user_type)
        fluent = problem.add_fluent(
            "fluent", IntType(), obj=user_type, default_initial_value=10
        )
        problem.add_object("o1", user_type)
        activity = problem.add_activity("activity")
        activity.set_fixed_duration(Plus(fluent(param), 10))

        problem.add_constraint(Equals(activity.start, 0))

        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert schedule.get(activity.start).constant_value() == 0
        assert schedule.get(activity.end).constant_value() == 20

    def test_activity_set_duration_bounds_with_parametric_fluents(
        self, problem: SchedulingProblem
    ):
        activity = problem.add_activity("activity", duration=5)

        user_type = UserType("user_type")
        param = activity.add_parameter("param", user_type)
        fluent = problem.add_fluent(
            "fluent", IntType(), obj=user_type, default_initial_value=10
        )
        problem.add_object("o1", user_type)

        activity.set_duration_bounds(lower=fluent(param), upper=Plus(fluent(param), 10))

        problem.add_constraint(Equals(activity.start, 0))

        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert schedule.get(activity.start).constant_value() == 0
        assert 10 <= schedule.get(activity.end).constant_value() <= 20

    def test_constraint_on_resource(self, problem: SchedulingProblem):
        problem.add_activity("activity", duration=1)
        resource = problem.add_resource("resource", capacity=2)

        problem.add_constraint(LE(resource, 2))

        self.problem_solved_satisficing_or_optimally(problem)

    def test_assign_effect(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=4)
        problem.set_initial_value(resource, 0)

        activity1 = problem.add_activity("activity1", duration=4)
        activity2 = problem.add_activity("activity2", duration=5)
        activity3 = problem.add_activity("activity3", duration=6)

        activity1.add_effect(activity1.end, resource, 2)

        activity2.add_decrease_effect(activity2.start, resource, 2)
        problem.add_increase_effect(Timing(0, activity2.end), resource, 4)

        activity3.add_decrease_effect(activity3.start, resource, 4)

        # should be activity1.end <= activity2.start <= activity2.end <= activity3.start
        problem.add_constraint(
            Not(
                And(
                    LE(activity1.end, activity2.start),
                    LE(activity2.end, activity3.start),
                )
            )
        )
        problem.add_constraint(LE(activity1.end, 100))
        problem.add_constraint(LE(activity2.end, 100))
        problem.add_constraint(LE(activity3.end, 100))

        self.problem_unsolvable(problem)

    def test_effect_with_time_expression(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=4)
        problem.set_initial_value(resource, 0)

        activity1 = problem.add_activity("activity1", duration=4)
        activity2 = problem.add_activity("activity2", duration=5)

        activity1.add_effect(Timing(5, activity1.start), resource, 2)
        activity2.uses(resource)

        problem.add_quality_metric(MinimizeMakespan())

        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert schedule.get(activity2.start).constant_value() == (
            schedule.get(activity1.start).constant_value() + 5 + 1
        )

    def test_effect_with_value_expression(self, problem: SchedulingProblem):
        variable = problem.add_variable(
            "variable", get_environment().type_manager.IntType()
        )
        problem.add_constraint(Equals(variable, 10))

        resource = problem.add_fluent(
            "resource",
            get_environment().type_manager.IntType(),
            default_initial_value=0,
        )
        problem.add_constraint(LE(0, resource))

        activity1 = problem.add_activity("activity1", duration=5)
        activity2 = problem.add_activity("activity2", duration=5)

        problem.add_effect(activity1.end, resource, Times(variable, 2))
        problem.add_decrease_effect(activity2.start, resource, 20)

        problem.add_constraint(LE(activity2.start, activity1.start))
        problem.add_constraint(LE(activity1.end, 11))
        problem.add_constraint(LE(activity2.end, 11))

        self.problem_unsolvable(problem)

    def test_effect_with_fluent_expression(self, problem: SchedulingProblem):
        resource1 = problem.add_fluent(
            "resource1",
            get_environment().type_manager.IntType(),
            default_initial_value=0,
        )
        resource2 = problem.add_fluent(
            "resource2",
            get_environment().type_manager.IntType(),
            default_initial_value=1,
        )

        activity1 = problem.add_activity("activity1", duration=5)
        activity2 = problem.add_activity("activity2", duration=5)
        problem.add_constraint(LE(activity1.end, activity2.start))

        problem.add_increase_effect(
            Timing(0, activity1.start), resource1, Times(resource2, 1)
        )
        problem.add_effect(activity1.end, resource1, Times(resource2, 2))
        activity2.uses(resource1, 2)

        self.problem_solved_satisficing_or_optimally(problem)

    def test_conditional_effects2(self, problem: SchedulingProblem):
        resource1 = problem.add_resource("resource1", capacity=2)
        resource2 = problem.add_resource("resource2", capacity=2)
        problem.set_initial_value(resource1, 0)
        problem.set_initial_value(resource2, 0)

        activity1 = problem.add_activity("activity1", duration=20)
        activity2 = problem.add_activity("activity2", duration=30)

        problem.add_increase_effect(
            Timing(0, activity1.start),
            resource1,
            1,
            condition=Equals(activity1.start, 10),
        )
        problem.add_decrease_effect(activity2.start, resource1, 1)

        problem.add_effect(
            activity1.end, resource2, 2, condition=Equals(activity1.end, 30)
        )
        problem.add_decrease_effect(activity2.end, resource2, 2)

        problem.add_quality_metric(MinimizeMakespan())

        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert schedule.get(activity1.start).constant_value() == 10

    def test_conditional_effects_with_fluent_expression(
        self, problem: SchedulingProblem
    ):
        fluent1 = problem.add_fluent("fluent1", IntType(), default_initial_value=1)
        fluent2 = problem.add_fluent(
            "fluent2", IntType(), default_initial_value=fluent1()
        )

        activity = problem.add_activity("activity", duration=20)

        problem.add_constraint(GE(fluent1, 0))
        problem.add_decrease_effect(
            activity.start, fluent1, Times(fluent2, 2), condition=LT(activity.start, 10)
        )
        problem.add_increase_effect(
            Timing(0, activity.start),
            fluent1,
            Times(fluent2, 2),
            condition=GE(activity.start, 10),
        )
        problem.add_quality_metric(MinimizeMakespan())

        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert schedule.get(activity.start).constant_value() == 10

    def test_condition_with_ClosedTimeInterval(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=10)
        resource = problem.add_resource("resource", capacity=2)
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )

        problem.add_condition(
            ClosedTimeInterval(Timing(0, activity.start), Timing(0, activity.end)),
            Equals(int_var, 5),
        )

        activity.add_decrease_effect(activity.start, resource, 1)
        activity.add_condition(
            ClosedTimeInterval(Timing(0, activity.start), Timing(0, activity.end)),
            Equals(resource, 1),
        )

        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert schedule.get(int_var).constant_value() == 5

    def test_condition_with_TimePointInterval(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=2)
        activity = problem.add_activity("activity", duration=10)
        activity.uses(resource, 1)
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )

        problem.add_condition(
            TimePointInterval(Timing(0, activity.start)),
            Equals(int_var, 5),
        )
        activity.add_condition(
            TimePointInterval(Timing(0, activity.start)),
            Equals(resource, 1),
        )
        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert schedule.get(int_var).constant_value() == 5

    def test_condition_forcing_equal_timepoint_values(self, problem: SchedulingProblem):
        activity1 = problem.add_activity("activity1", duration=10)
        activity2 = problem.add_activity("activity2", duration=10)
        activity3 = problem.add_activity("activity3", duration=10)
        resource = problem.add_resource("resource", capacity=3)

        # activity1.start == activity2.start == activity3.start == 0
        problem.add_constraint(Equals(activity1.start, 0))
        problem.add_constraint(Equals(activity2.start, 0))
        problem.add_constraint(Equals(activity3.start, 0))

        activity1.add_decrease_effect(activity1.start, resource, 1)
        activity2.add_decrease_effect(activity2.start, resource, 1)
        activity3.add_decrease_effect(activity3.start, resource, 1)

        activity1.add_condition(
            ClosedTimeInterval(Timing(0, activity1.start), Timing(0, activity1.start)),
            Equals(resource, 0),
        )

        self.problem_solved_satisficing_or_optimally(problem)

    def test_condition_without_resources(self, problem: SchedulingProblem):
        activity1 = problem.add_activity("activity1", duration=10)
        activity2 = problem.add_activity("activity2", duration=10)
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )

        problem.add_constraint(LT(activity1.end, activity2.start))

        # these two conditions act as constraints since no resources are involved
        problem.add_condition(
            ClosedTimeInterval(Timing(0, activity1.start), Timing(0, activity1.end)),
            Equals(int_var, 5),
        )
        problem.add_condition(
            ClosedTimeInterval(Timing(0, activity2.start), Timing(0, activity2.end)),
            Equals(int_var, 4),
        )
        self.problem_unsolvable(problem)

    def test_condition_with_invalid_interval(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=10)
        resource = problem.add_resource("resource", capacity=2)
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )

        # this condition should not be applied since the interval is [activity.end,
        # activity.start]
        problem.add_condition(
            ClosedTimeInterval(Timing(0, activity.end), Timing(0, activity.start)),
            Equals(int_var, 5),
        )
        problem.add_constraint(Equals(int_var, 4))

        # this condition should not be applied since the interval is [activity.end,
        # activity.start]
        activity.add_condition(
            ClosedTimeInterval(Timing(0, activity.end), Timing(0, activity.start)),
            Equals(resource, 1),
        )

        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert schedule.get(int_var).constant_value() == 4

    def test_int_fluents(self, problem: SchedulingProblem):
        busy = problem.add_fluent(
            "busy", get_environment().type_manager.IntType(), default_initial_value=1
        )
        problem.set_initial_value(busy, 0)

        activity1 = problem.add_activity("activity1", duration=5)
        activity1.add_condition(activity1.start, Equals(busy, 0))
        activity1.add_effect(activity1.start + 1, busy, 1)
        activity1.add_effect(activity1.end, busy, 0)

        activity2 = problem.add_activity("activity2", duration=5)
        activity2.add_condition(activity2.start, Equals(busy, 0))
        activity2.add_effect(activity2.start + 1, busy, 1)
        activity2.add_effect(activity2.end, busy, 0)

        problem.add_constraint(LE(activity2.end, activity1.start))

        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert (
            schedule.get(activity2.end).constant_value()
            <= schedule.get(activity1.start).constant_value()
        )

    def test_bool_fluents(self, problem: SchedulingProblem):
        busy = problem.add_fluent("busy", default_initial_value=True)
        problem.set_initial_value(busy, False)

        activity1 = problem.add_activity("activity1", duration=5)
        activity1.add_condition(activity1.start, Not(busy))
        activity1.add_effect(activity1.start + 1, busy, True)
        activity1.add_effect(activity1.end, busy, False)

        activity2 = problem.add_activity("activity2", duration=5)
        activity2.add_condition(activity2.start, Not(busy))
        activity2.add_effect(activity2.start + 1, busy, True)
        activity2.add_effect(activity2.end, busy, False)

        problem.add_constraint(LE(activity2.end, activity1.start))

        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert (
            schedule.get(activity2.end).constant_value()
            <= schedule.get(activity1.start).constant_value()
        )

    def test_fluents_with_parameters(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)

        user_type_parent = UserType("user_type_parent")
        user_type1 = UserType("user_type1", user_type_parent)
        user_type2 = UserType("user_type2", user_type_parent)
        user_type3 = UserType("user_type3")
        param1 = activity.add_parameter("param1", user_type1)
        param2 = activity.add_parameter("param2", user_type2)
        fluent1 = problem.add_fluent(
            "fluent1",
            IntType(),
            obj1=user_type_parent,
            obj2=user_type2,
            default_initial_value=2,
        )
        fluent2 = problem.add_fluent(
            "fluent2",
            IntType(),
            obj=user_type1,
            default_initial_value=0,
        )
        fluent3 = problem.add_fluent(
            "fluent3", BoolType(), obj=user_type3, default_initial_value=False
        )
        o1 = problem.add_object("o1", user_type1)
        o2 = problem.add_object("o2", user_type1)
        problem.add_object("o3", user_type2)
        problem.add_object("o4", user_type2)
        problem.add_object("o5", user_type_parent)
        o6 = problem.add_object("o6", user_type3)

        activity.add_increase_effect(activity.end, fluent1(param1, param2), 2)
        activity.add_effect(activity.start, fluent1(param1, param2), 2)
        activity.add_increase_effect(activity.end, fluent2(o1), 1)
        activity.add_effect(activity.start, fluent3(o6), True)

        problem.add_constraint(Equals(param1, o2))

        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert schedule.get(param1).constant_value() == o2

    def test_fluents_with_parameters_and_conditions(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)

        user_type1 = UserType("user_type1")
        user_type2 = UserType("user_type2")
        param1 = activity.add_parameter("param1", user_type1)
        param2 = activity.add_parameter("param2", user_type2)
        param3 = activity.add_parameter("param3", user_type2)
        fluent = problem.add_fluent(
            "fluent",
            IntType(),
            obj1=user_type1,
            obj2=user_type2,
            default_initial_value=2,
        )
        problem.add_object("o1", user_type1)
        o2 = problem.add_object("o2", user_type1)
        o3 = problem.add_object("o3", user_type2)
        problem.add_object("o4", user_type2)

        final_value_is_6 = problem.add_variable("final_value_is_6", BoolType())

        activity.add_increase_effect(activity.start, fluent(param1, param2), 2)
        activity.add_increase_effect(activity.end, fluent(param1, param3), 4)
        activity.add_decrease_effect(activity.end, fluent(param1, param3), 2)

        problem.add_constraint(Equals(param1, o2))
        problem.add_constraint(Equals(param2, o3))
        problem.add_constraint(Equals(param3, o3))

        problem.add_condition(
            TimePointInterval(Timing(0, activity.start)),
            Equals(fluent(param1, param2), 4),
        )
        problem.add_condition(
            TimePointInterval(Timing(0, activity.start)),
            Equals(fluent(param1, param3), 4),
        )
        problem.add_condition(
            ClosedTimeInterval(Timing(0, activity.start), Timing(0, activity.end)),
            And(GE(fluent(param1, param2), 2), LE(fluent(param1, param2), 6)),
        )

        problem.add_constraint(
            Iff(GT(fluent(param1, param2), 6), Not(final_value_is_6))
        )

        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert schedule.get(param1).constant_value() == o2
        assert schedule.get(param2).constant_value() == o3
        assert schedule.get(param3).constant_value() == o3
        assert schedule.get(final_value_is_6).constant_value() is True

    def test_constraint_and_condition_with_multiple_parametric_fluents(
        self, problem: SchedulingProblem
    ):
        activity = problem.add_activity("activity", duration=5)

        user_type1 = UserType("user_type1")
        user_type2 = UserType("user_type2")
        param1 = activity.add_parameter("param1", user_type1)
        param2 = activity.add_parameter("param2", user_type2)
        fluent1 = problem.add_fluent(
            "fluent1",
            IntType(),
            obj1=user_type1,
            obj2=user_type2,
            default_initial_value=0,
        )
        fluent2 = problem.add_fluent(
            "fluent2",
            IntType(),
            obj=user_type1,
            default_initial_value=0,
        )
        o1 = problem.add_object("o1", user_type1)
        problem.add_object("o2", user_type1)
        o3 = problem.add_object("o3", user_type2)
        problem.add_object("o4", user_type2)

        problem.add_constraint(Equals(param1, o1))
        problem.add_constraint(Equals(param2, o3))

        activity.add_increase_effect(activity.start, fluent1(param1, param2), 1)
        activity.add_increase_effect(activity.start, fluent2(param1), 1)
        activity.add_increase_effect(activity.end, fluent1(param1, param2), 2)
        activity.add_increase_effect(activity.end, fluent2(param1), 2)

        problem.add_constraint(Equals(fluent1(param1, param2), fluent2(param1)))

        problem.add_condition(
            TimePointInterval(Timing(0, activity.start)),
            And(Equals(fluent1(param1, param2), 1), Equals(fluent2(param1), 1)),
        )
        problem.add_condition(
            ClosedTimeInterval(Timing(0, activity.start), Timing(0, activity.end)),
            And(
                And(GE(fluent1(o1, o3), 1), LE(fluent1(param1, param2), 3)),
                And(GE(fluent2(o1), 1), LE(fluent2(param1), 3)),
            ),
        )

        schedule = self.problem_solved_satisficing_or_optimally(problem)
        assert schedule.get(param1).constant_value() == o1
        assert schedule.get(param2).constant_value() == o3

    def test_effects_with_parametric_fluents(self, problem: SchedulingProblem):
        activity1 = problem.add_activity("activity1", duration=5)
        activity2 = problem.add_activity("activity2", duration=5)

        user_type = UserType("user_type")
        param = activity1.add_parameter("param", user_type)
        fluent = problem.add_fluent(
            "fluent", IntType(), obj=user_type, default_initial_value=0
        )
        o1 = problem.add_object("o1", user_type)
        o2 = problem.add_object("o2", user_type)

        bool_var = problem.add_variable("bool_var", BoolType())

        activity1.add_increase_effect(activity2.start, fluent(param), 1)
        activity1.add_increase_effect(
            activity2.start, fluent(param), 6, condition=Not(bool_var)
        )
        activity1.add_increase_effect(
            activity2.start, fluent(param), 7, condition=bool_var
        )

        activity1.add_increase_effect(activity1.start, fluent(o1), 2)
        activity1.add_increase_effect(
            activity1.start, fluent(o1), 3, condition=bool_var
        )

        activity1.add_increase_effect(
            activity1.start, fluent(o2), 4, condition=bool_var
        )

        problem.add_constraint(Equals(activity1.start, activity2.start))
        problem.add_constraint(bool_var)
        problem.add_constraint(Equals(param, o1))

        problem.add_constraint(GE(fluent(o1), 0))
        problem.add_constraint(LE(fluent(o1), 13))
        problem.add_constraint(GE(fluent(o2), 0))
        problem.add_constraint(LE(fluent(o2), 4))

        self.problem_solved_satisficing_or_optimally(problem)

    def test_parametric_fluents_in_effect_value(self, problem: SchedulingProblem):
        activity1 = problem.add_activity("activity1", duration=5)
        activity2 = problem.add_activity("activity2", duration=5)

        user_type = UserType("user_type")
        param = activity1.add_parameter("param", user_type)
        fluent1 = problem.add_fluent(
            "fluent1", IntType(), obj=user_type, default_initial_value=0
        )
        fluent2 = problem.add_fluent(
            "fluent2", IntType(), obj=user_type, default_initial_value=2
        )
        o1 = problem.add_object("o1", user_type)
        problem.add_object("o2", user_type)

        activity1.add_increase_effect(
            activity2.start, fluent1(param), Times(fluent2(param), 2)
        )

        problem.add_constraint(Equals(param, o1))
        problem.add_constraint(LT(fluent1(o1), 4))

        self.problem_unsolvable(problem)

    def test_set_fluent_non_constant_initial_value(self, problem: SchedulingProblem):
        variable = problem.add_variable(
            "variable", get_environment().type_manager.IntType()
        )
        problem.add_constraint(Equals(variable, 1))

        resource1 = problem.add_fluent(
            "resource1",
            get_environment().type_manager.IntType(lower_bound=-1, upper_bound=11),
            default_initial_value=(
                problem.environment.expression_manager.ParameterExp(variable)
            ),
        )
        problem.add_constraint(LE(0, resource1))

        resource2 = problem.add_fluent(
            "resource2",
            get_environment().type_manager.IntType(),
        )
        problem.set_initial_value(resource2, Times(variable, 2))
        problem.add_constraint(LE(0, resource2))

        activity = problem.add_activity("activity", duration=5)
        activity.add_decrease_effect(activity.start, resource1, 1)
        activity.add_decrease_effect(activity.start, resource2, 2)
        activity.add_constraint(Equals(activity.start, 0))

        self.problem_solved_satisficing_or_optimally(problem)

    def test_uninitialized_fluent(self, problem: SchedulingProblem):
        problem.add_fluent(
            "fluent",
            get_environment().type_manager.IntType(lower_bound=-1, upper_bound=11),
        )
        problem.add_activity("activity", 5)
        self.problem_solved_satisficing_or_optimally(problem)

    def test_set_fluent_initial_value_as_fluent_exp(self, problem: SchedulingProblem):
        fluent1 = problem.add_fluent(
            "fluent1",
            IntType(lower_bound=0),
            default_initial_value=1,
        )
        fluent2 = problem.add_fluent(
            "fluent2",
            IntType(lower_bound=0),
            default_initial_value=Times(fluent1, 2),
        )
        user_type = UserType("user_type")
        fluent3 = problem.add_fluent(
            "fluent3", IntType(), obj=user_type, default_initial_value=Times(fluent2, 2)
        )
        o1 = problem.add_object("o1", user_type)

        activity = problem.add_activity("activity", duration=5)
        activity.add_decrease_effect(activity.start, fluent1, 1)
        activity.add_decrease_effect(activity.start, fluent2, 2)
        activity.add_decrease_effect(activity.start, fluent3(o1), 2)

        self.problem_solved_satisficing_or_optimally(problem)

    def test_set_fluent_initial_value_as_parametric_fluent_exp(
        self, problem: SchedulingProblem
    ):
        user_type = UserType("user_type")
        param = problem.add_variable("param", user_type)
        o1 = problem.add_object("o1", user_type)

        fluent1 = problem.add_fluent(
            "fluent1",
            IntType(lower_bound=0),
            obj=user_type,
            default_initial_value=1,
        )
        fluent2 = problem.add_fluent(
            "fluent2",
            IntType(lower_bound=0),
            default_initial_value=Times(fluent1(param), 2),
        )
        fluent3 = problem.add_fluent(
            "fluent3", IntType(), obj=user_type, default_initial_value=Times(fluent2, 2)
        )

        problem.add_activity("activity", duration=5)

        problem.add_constraint(Equals(fluent1(o1), 1))
        problem.add_constraint(Equals(fluent2, 2))
        problem.add_constraint(Equals(fluent3(o1), 4))

        self.problem_solved_satisficing_or_optimally(problem)

    def test_constraint_fluents_with_args(self, problem: SchedulingProblem):
        user_type = UserType("user_type")
        fluent = problem.add_fluent(
            "fluent",
            IntType(lower_bound=0, upper_bound=10),
            obj=user_type,
        )
        o1 = problem.add_object("o1", user_type)
        o2 = problem.add_object("o2", user_type)

        activity = problem.add_activity("activity", 2)
        activity.add_constraint(Equals(fluent(o1), 2))
        activity.add_constraint(Equals(fluent(o2), 3))

        self.problem_solved_satisficing_or_optimally(problem)

    def test_condition_fluents_with_args(self, problem: SchedulingProblem):
        user_type = UserType("user_type")
        fluent = problem.add_fluent(
            "fluent", IntType(lower_bound=0, upper_bound=10), obj=user_type
        )
        o1 = problem.add_object("o1", user_type)
        o2 = problem.add_object("o2", user_type)
        # problem.set_initial_value(fluent(o1), 2)
        # problem.set_initial_value(fluent(o2), 3)

        activity = problem.add_activity("activity", 10)
        activity.add_condition(
            ClosedTimeInterval(Timing(0, activity.start), Timing(0, activity.end)),
            Equals(fluent(o1), 2),
        )
        activity.add_condition(
            ClosedTimeInterval(Timing(0, activity.end), Timing(0, activity.end)),
            Equals(fluent(o2), 3),
        )

        self.problem_solved_satisficing_or_optimally(problem)

    def test_not_supported_fluents_with_different_params_at_same_timepoint(
        self, problem: SchedulingProblem
    ):
        activity = problem.add_activity("activity", duration=5)

        user_type1 = UserType("user_type1")
        user_type2 = UserType("user_type2")
        param1 = activity.add_parameter("param1", user_type1)
        param2 = activity.add_parameter("param2", user_type2)
        param3 = activity.add_parameter("param3", user_type2)
        fluent = problem.add_fluent(
            "fluent", IntType(), obj1=user_type1, obj2=user_type2
        )
        problem.add_object("o1", user_type1)
        problem.add_object("o2", user_type2)

        activity.add_increase_effect(activity.start, fluent(param1, param2), 2)
        activity.add_increase_effect(activity.start, fluent(param1, param3), 2)

        self.problem_unsupported(problem)

    def test_not_supported_same_fluent_in_effect_value(
        self, problem: SchedulingProblem
    ):
        activity1 = problem.add_activity("activity1", duration=5)
        activity2 = problem.add_activity("activity2", duration=5)

        user_type = UserType("user_type")
        param = activity1.add_parameter("param", user_type)
        fluent = problem.add_fluent(
            "fluent", IntType(), obj=user_type, default_initial_value=0
        )
        o1 = problem.add_object("o1", user_type)

        activity1.add_increase_effect(activity2.start, fluent(param), fluent(o1))
        self.problem_unsupported(problem)
