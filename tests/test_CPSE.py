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

from unified_planning.model.scheduling import SchedulingProblem
from unified_planning.shortcuts import *

from cpse import CPSE

# `problem` is a pytest fixture; importing it here registers it for this module.
from .CommonTests import CommonTests, problem  # noqa: F401


class TestCPSE(CommonTests):
    def engine_name(self):
        return "cpse"

    def engine_class(self):
        return CPSE

    def test_optional_activities(self, problem: SchedulingProblem):
        activity1 = problem.add_activity("activity1", 10, optional=True)
        activity2 = problem.add_activity("activity2", 3, optional=True)

        problem.add_constraint(
            Or(
                And(activity1.present, Not(activity2.present)),
                And(activity2.present, Not(activity1.present)),
            )
        )

        problem.add_quality_metric(MinimizeMakespan())

        res = self.problem_solved_satisficing_or_optimally(problem)
        present1 = res.plan.get(activity1.present.presence()).constant_value()
        present2 = res.plan.get(activity2.present.presence()).constant_value()
        assert present2 and not present1

    def test_constraints_with_optional_activities(self, problem: SchedulingProblem):
        activity1 = problem.add_activity("activity1", 10, optional=True)
        activity2 = problem.add_activity("activity2", 3, optional=True)
        int_var = problem.add_variable("int_var", IntType())
        bool_var = problem.add_variable("bool_var", BoolType())

        problem.add_constraint(
            Or(
                And(activity1.present, Not(activity2.present)),
                And(activity2.present, Not(activity1.present)),
            )
        )
        problem.add_constraint(activity1.present)

        problem.add_constraint(bool_var, scope=[activity1.present])
        problem.add_constraint(
            Not(bool_var), scope=[activity1.present, activity2.present]
        )

        activity1.add_constraint(LT(int_var, 5))
        activity2.add_constraint(GT(int_var, 5))

        res = self.problem_solved_satisficing_or_optimally(problem)

        present1 = res.plan.get(activity1.present.presence()).constant_value()
        present2 = res.plan.get(activity2.present.presence()).constant_value()
        assert present1 and not present2
        assert res.plan.get(int_var).constant_value() < 5
        assert res.plan.get(bool_var).constant_value()

    def test_effects_with_optional_activities(self, problem: SchedulingProblem):
        activity1 = problem.add_activity("activity1", 10, optional=True)
        activity2 = problem.add_activity("activity2", 3, optional=True)
        fluent = problem.add_fluent(
            "fluent", IntType(lower_bound=0, upper_bound=1), default_initial_value=0
        )

        problem.add_constraint(
            Or(
                And(activity1.present, Not(activity2.present)),
                And(activity2.present, Not(activity1.present)),
            )
        )
        problem.add_constraint(activity2.present)

        activity1.add_increase_effect(activity1.start + 1, fluent, 1)
        activity2.add_increase_effect(activity2.start, fluent, 1)

        self.problem_solved_satisficing_or_optimally(problem)

    def test_conditions_with_optional_activities(self, problem: SchedulingProblem):
        activity1 = problem.add_activity("activity1", 10, optional=True)
        activity2 = problem.add_activity("activity2", 3, optional=True)
        int_var = problem.add_variable("int_var", IntType())

        problem.add_constraint(
            Or(
                And(activity1.present, Not(activity2.present)),
                And(activity2.present, Not(activity1.present)),
            )
        )
        problem.add_constraint(activity1.present)

        activity1.add_condition(
            ClosedTimeInterval(Timing(3, activity1.start), Timing(-2, activity1.end)),
            LT(int_var, 5),
        )

        activity2.add_condition(
            ClosedTimeInterval(Timing(3, activity1.start), Timing(-2, activity1.end)),
            GT(int_var, 5),
        )

        res = self.problem_solved_satisficing_or_optimally(problem)

        present1 = res.plan.get(activity1.present.presence()).constant_value()
        present2 = res.plan.get(activity2.present.presence()).constant_value()
        assert present1 and not present2
        assert res.plan.get(int_var).constant_value() < 5

    def test_not_supported_assign_effect(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)
        resource = problem.add_resource("resource", 10)
        problem.add_effect(activity.start, resource, 5)

        self.problem_unsupported(problem)

    def test_not_supported_effect_value(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=2)
        variable = problem.add_variable(
            "variable", get_environment().type_manager.IntType()
        )
        problem.add_constraint(Equals(variable, 1))
        activity = problem.add_activity("activity", duration=20)
        activity.add_decrease_effect(activity.start, resource, variable)

        self.problem_unsupported(problem)

    def test_not_supported_fluent_exp(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", 10)
        problem.add_constraint(LT(1, resource))

        self.problem_unsupported(problem)

    def test_not_supported_condition_with_fluent_exp(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)
        resource = problem.add_resource("resource", 10)
        problem.add_constraint(LT(1, resource))
        problem.add_condition(
            ClosedTimeInterval(activity.start, activity.end), LT(1, resource)
        )

        self.problem_unsupported(problem)

    def test_not_supported_uninitialized_fluent(self, problem: SchedulingProblem):
        fluent = problem.add_fluent(
            "fluent",
            get_environment().type_manager.IntType(lower_bound=-1, upper_bound=11),
        )
        activity = problem.add_activity("activity", 5)
        problem.add_increase_effect(activity.start, fluent, 1)
        self.problem_unsupported(problem)

    def test_not_supported_fluent_initial_value_as_fluent_exp(
        self, problem: SchedulingProblem
    ):
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

        activity = problem.add_activity("activity", duration=5)
        activity.add_decrease_effect(activity.start, fluent1, 1)
        activity.add_decrease_effect(activity.start, fluent2, 2)

        self.problem_unsupported(problem)

    def test_not_supported_fluent_with_parameters(self, problem: SchedulingProblem):
        user_type = UserType("user_type")
        o1 = problem.add_object("o1", user_type)
        o2 = problem.add_object("o2", user_type)
        parameter = problem.add_variable("parameter", user_type)
        fluent = problem.add_fluent(
            "fluent", IntType(), obj=user_type, default_initial_value=0
        )
        problem.set_initial_value(fluent(parameter), 1)

        activity = problem.add_activity("activity", 2)
        activity.add_increase_effect(activity.start, fluent(parameter), 1)

        self.problem_unsupported(problem)
