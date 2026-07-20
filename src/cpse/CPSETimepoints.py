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

import itertools
from collections import defaultdict
from collections.abc import Iterable
from typing import cast

import unified_planning as up
from ortools.sat.python import cp_model
from unified_planning.model import (
    Effect,
    Fluent,
    FNode,
    Object,
    Parameter,
    ProblemKind,
    timing,
)
from unified_planning.model.scheduling import Activity, SchedulingProblem

from .CPSEBaseEngine import CPSEBaseEngine


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

    def solve_init(self):
        super().solve_init()

        self.timepoints: list[cp_model.IntVar] = []
        self.assignment_matrix: dict[
            tuple[timing.Timepoint, int], list[cp_model.IntVar]
        ] = {}
        self.resources: dict[FNode, list[list[cp_model.IntVar]]] = {}
        self.parametric_fluent_assignments: dict[
            FNode, dict[FNode, tuple[tuple[Object, ...], cp_model.IntVar]]
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
        return bool(problem_kind <= CPSETimepoints.supported_kind())

    def check_if_supported_problem(self, problem: "up.model.AbstractProblem"):
        """
        Checks if the given problem is a supported instance of `SchedulingProblem`.
        Raises `NotImplementedError` if unsupported.

        Args:
            problem (up.model.AbstractProblem): The problem instance to validate.

        Raises:
            NotImplementedError: If the problem is not supported.
        """

        if not isinstance(problem, SchedulingProblem):
            raise NotImplementedError(f"Problem of type {type(problem)} not supported.")
        problem: SchedulingProblem

        if not problem.discrete_time:
            raise NotImplementedError("Continuous time not supported.")

        fluents: dict[timing.Timing, dict[Fluent, FNode]] = defaultdict(dict)
        contains_parameters: dict[timing.Timing, dict[Fluent, bool]] = defaultdict(dict)
        for effect_timing, eff, _activity in problem.all_effects():
            fluent = eff.fluent.fluent()
            contains_params = self._fluent_exp_contains_parameters(eff.fluent)
            if fluent in contains_parameters[effect_timing] and (
                contains_parameters[effect_timing][fluent] != contains_params
                or (
                    contains_parameters[effect_timing][fluent]
                    and fluents[effect_timing][fluent] != eff.fluent
                )
            ):
                raise NotImplementedError(
                    "Fluents with parameters cannot coexist with non-parametric "
                    "fluents at the same timepoint."
                )
            fluents[effect_timing][fluent] = eff.fluent
            contains_parameters[effect_timing][fluent] = contains_params

        for _effect_timing, eff, _activity in problem.all_effects():
            fluent = eff.fluent.fluent()
            all_fluent_exps = self.extract_all_fluent_exp_from_fnode(eff.value)
            for fluent_exp in all_fluent_exps:
                if fluent == fluent_exp.fluent():
                    raise NotImplementedError(
                        "The fluent affected by an effect cannot be included in the "
                        "effect's value."
                    )

    def _add_activity_timepoints(
        self, activity: Activity
    ) -> tuple[cp_model.IntVar, int | cp_model.IntVar, cp_model.IntVar]:
        """
        Adds variables for the start, end, and duration of an activity, and enforces
        constraints on them.

        Args:
            activity (Activity): The activity for which variables are being defined.

        Returns:
            Tuple[Union[int, cp_model.IntVar], Union[int, cp_model.IntVar],
                Union[int, cp_model.IntVar]]:
                A tuple containing the start time, end time, and duration for the
                activity,
                each represented as either an integer (for fixed values) or a model
                variable.
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
        duration_var: int | cp_model.IntVar
        if lower.is_int_constant() and upper.is_int_constant():
            lower_val = lower.int_constant_value()
            upper_val = upper.int_constant_value()
            if lower_val == upper_val:
                # FixedDuration
                duration_var = upper_val
            else:
                # ClosedDurationInterval
                duration_var = self.model.new_int_var(
                    self.lower_bound, self.upper_bound, "duration_" + activity.name
                )
                self.model.add_linear_constraint(duration_var, lower_val, upper_val)
        else:
            duration_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "duration_" + activity.name
            )

            def constrain_duration():
                self._set_fluent_vars_at_timepoint(tp_idx=-1)
                if lower != upper:
                    all_fluent_exps = list(
                        self.extract_all_parametric_fluent_exp_from_fnode(lower).union(
                            self.extract_all_parametric_fluent_exp_from_fnode(upper)
                        )
                    )
                else:
                    all_fluent_exps = list(
                        self.extract_all_parametric_fluent_exp_from_fnode(lower)
                    )
                all_parameters = list(
                    self.extract_all_params_from_fluent_exps(all_fluent_exps)
                )
                if len(all_parameters) == 0:
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
                else:
                    for (
                        ground_fluent_exps,
                        fluent_assignment_vars,
                    ) in self.get_all_fluent_assignments(
                        all_fluent_exps, all_parameters
                    ):
                        self._set_parametric_fluent_vars_at_timepoint(
                            all_fluent_exps, ground_fluent_exps, tp_idx=-1
                        )

                        if lower == upper:
                            # FixedDuration
                            duration_exp = self.fnode_to_value_or_variable(upper)
                            self.model.add(
                                duration_var == duration_exp
                            ).only_enforce_if(fluent_assignment_vars)
                        else:
                            # ClosedDurationInterval
                            lower_exp = self.fnode_to_value_or_variable(lower)
                            upper_exp = self.fnode_to_value_or_variable(upper)
                            self.model.add(lower_exp <= duration_var).only_enforce_if(
                                fluent_assignment_vars
                            )
                            self.model.add(duration_var <= upper_exp).only_enforce_if(
                                fluent_assignment_vars
                            )

            self._postponed_constraints.append(constrain_duration)

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
                stack.extend(fnode.args)

    def _collect_all_problem_timings(
        self, problem: SchedulingProblem
    ) -> list[tuple["timing.Timepoint", int]]:
        """
        Collects all unique timepoints and their associated delays from the given
        scheduling problem.

        Args:
            problem (SchedulingProblem): The scheduling problem from which to extract
                timing information.

        Returns:
            List[Tuple["timing.Timepoint", int]]:
                A list of unique timepoints and their associated delays, represented as
                tuples
                of the form `(timepoint, delay)`.
        """

        problem_timings: set[tuple[timing.Timepoint, int]] = set()

        for activity in problem.activities:
            problem_timings.add((activity.start, 0))
            problem_timings.add((activity.end, 0))

        for fnode in problem.all_constraints():
            for fnode_timing in self._fnode_timings(fnode):
                assert isinstance(fnode_timing.delay, int)
                problem_timings.add((fnode_timing.timepoint, fnode_timing.delay))

        for time_interval, fnode, _activity in problem.all_conditions():
            assert isinstance(time_interval.lower.delay, int)
            problem_timings.add(
                (time_interval.lower.timepoint, time_interval.lower.delay)
            )
            assert isinstance(time_interval.upper.delay, int)
            problem_timings.add(
                (time_interval.upper.timepoint, time_interval.upper.delay)
            )
            for fnode_timing in self._fnode_timings(fnode):
                assert isinstance(fnode_timing.delay, int)
                problem_timings.add((fnode_timing.timepoint, fnode_timing.delay))

        for effect_timing, effect, _activity in problem.all_effects():
            assert isinstance(effect_timing.delay, int)
            problem_timings.add((effect_timing.timepoint, effect_timing.delay))
            if effect.is_conditional():
                for fnode_timing in self._fnode_timings(effect.condition):
                    assert isinstance(fnode_timing.delay, int)
                    problem_timings.add((fnode_timing.timepoint, fnode_timing.delay))

        return list(problem_timings)

    def get_all_fluent_expressions(self, problem: SchedulingProblem) -> Iterable[FNode]:
        """
        Retrieves all fluent expressions from the given scheduling problem.

        Args:
            problem (SchedulingProblem): The scheduling problem from which to extract
                fluent expressions.

        Yields:
            FNode: Each fluent expression.
        """

        for fluent in problem.fluents:
            if fluent.arity == 0:
                yield fluent()
                continue

            domains = [
                self._get_compatible_objects(arg.type)
                + self._get_compatible_parameters(arg.type)
                for arg in fluent.signature
            ]

            for args in itertools.product(*domains):
                yield fluent(*args)

    def get_all_ground_fluent_expressions(
        self, problem: SchedulingProblem
    ) -> Iterable[FNode]:
        """
        Retrieves all ground fluent expressions from the given scheduling problem.

        Args:
            problem (SchedulingProblem): The scheduling problem from which to extract
                fluent expressions.

        Yields:
            FNode: Each ground fluent expression.
        """

        for fluent in problem.fluents:
            if fluent.arity == 0:
                yield fluent()
                continue

            domains = [
                self._get_compatible_objects(arg.type) for arg in fluent.signature
            ]

            for args in itertools.product(*domains):
                yield fluent(*args)

    def get_all_fluent_assignments(
        self, fluent_exps: list[FNode], params: list[Parameter]
    ) -> Iterable[tuple[list[FNode], list[cp_model.IntVar]]]:
        """
        Generates all possible grounded assignments of parameters to fluent expressions.

        Args:
            fluent_exps (List[FNode]): A list of parametrized fluent expressions.
            params (List[Parameter]): A list of parameters for which compatible
                assignments
                will be generated.

        Returns:
            Iterable[Tuple[List[FNode], List[cp_model.IntVar]]]: An iterable of tuples,
                where:
                - The first element is a list of `FNode` objects representing
                  grounded fluent expressions.
                - The second element is a list of `cp_model.IntVar` objects representing
                the
                corresponding model variables for the assignments.
        """

        domains = [self._get_compatible_objects(param.type) for param in params]
        params_indexes = {p: i for i, p in enumerate(params)}

        for objs in itertools.product(*domains):
            ground_fluent_exps = []
            fluent_assignment_vars = []
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
                ground_fluent_exps.append(ground_fluent_exp)
                fluent_assignment_vars.append(assignment_var)
            yield (ground_fluent_exps, fluent_assignment_vars)

    def all_parameters_used_in_fluents(
        self, problem: SchedulingProblem
    ) -> set[Parameter]:
        """
        Identifies all parameters used in the fluent expressions of the given problem.

        Args:
            problem (SchedulingProblem): The scheduling problem containing fluent
                expressions.

        Returns:
            Set[Parameter]: A set of `Parameter` objects that are used in the fluent
                expressions.
        """

        all_parameters = set()
        for fluent_exp in self.get_all_fluent_expressions(problem):
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

    def _set_parametric_fluent_vars_at_timepoint(
        self,
        parametric_fluent_exps: list[FNode],
        ground_fluent_exps: list[FNode],
        tp_idx: int,
    ):
        """
        Associates parametric fluent expressions with their corresponding ground fluents
        at a specific timepoint in the model variables.

        Args:
            parametric_fluent_exps (List[FNode]): A list of parametric fluent
                expressions
                that need to be mapped to ground fluent expressions.
            ground_fluent_exps (List[FNode]): A list of ground fluent expressions
                corresponding to the parametric fluents.
            tp_idx (int): The timepoint index used to associate each fluent with
                its corresponding value at the given timepoint. Must satisfy
                `-1 <= tp_idx < len(self.timepoints)`.
        """

        assert -1 <= tp_idx < len(self.timepoints)
        for fluent_exp, ground_fluent_exp in zip(
            parametric_fluent_exps, ground_fluent_exps, strict=True
        ):
            self._model_vars[fluent_exp] = self.resources[ground_fluent_exp][
                tp_idx + 1
            ][-1]

    def are_timepoints_equal(self, i: int, j: int) -> cp_model.IntVar:
        """
        Determines if the values of two timepoints are equal in the model.

        This method creates and returns a boolean variable (`cp_model.IntVar`) that
        represents
        whether the values of two specified timepoints are equal.

        Args:
            i (int): The index of the first timepoint.
            j (int): The index of the second timepoint.

        Returns:
            cp_model.IntVar: A boolean variable that evaluates to `True` if the two
                timepoints
            are equal, otherwise `False`.
        """

        assert i != j
        assert i < len(self.timepoints)
        assert j < len(self.timepoints)

        tp_i_j_are_equal_key = f"{self.timepoints[i].name} == {self.timepoints[j].name}"
        tp_j_i_are_equal_key = f"{self.timepoints[j].name} == {self.timepoints[i].name}"
        if tp_i_j_are_equal_key not in self._variables_cache:
            are_timepoints_equal = self.new_bool_var()
            self.model.add(self.timepoints[i] == self.timepoints[j]).only_enforce_if(
                are_timepoints_equal
            )
            self.model.add(self.timepoints[i] != self.timepoints[j]).only_enforce_if(
                are_timepoints_equal.negated()
            )
            self._variables_cache[tp_i_j_are_equal_key] = are_timepoints_equal
            self._variables_cache[tp_j_i_are_equal_key] = are_timepoints_equal

        return self._variables_cache[tp_i_j_are_equal_key]

    def timepoints_setup(self, problem: SchedulingProblem):
        """
        Defines the constraints that links each activity timepoint (start or end)
        to exactly one timepoint in the `timepoints` list.

        Args:
            problem (SchedulingProblem): The scheduling problem.
        """

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
            for _ in range(len(self.timepoints)):
                self.resources[fluent_exp].append([])

        def constrain_fluent_initial_values():
            self._set_fluent_vars_at_timepoint(tp_idx=-1)
            for fluent_exp in self.resources:
                if fluent_exp not in self._fluent_initial_value:
                    # uninitialized fluent
                    continue

                init_value = self._fluent_initial_value[fluent_exp]
                all_fluent_exps = list(
                    self.extract_all_parametric_fluent_exp_from_fnode(init_value)
                )
                all_parameters = list(
                    self.extract_all_params_from_fluent_exps(all_fluent_exps)
                )
                if len(all_parameters) == 0:
                    value_var = self.fnode_to_value_or_variable(init_value)
                    self.model.add(self.resources[fluent_exp][0][0] == value_var)
                else:
                    # initial value fnode contains fluents with parameters
                    for (
                        ground_fluent_exps,
                        fluent_assignment_vars,
                    ) in self.get_all_fluent_assignments(
                        all_fluent_exps, all_parameters
                    ):
                        for fluent_exp_prime, ground_fluent_exp in zip(
                            all_fluent_exps, ground_fluent_exps, strict=True
                        ):
                            self._model_vars[fluent_exp_prime] = self.resources[
                                ground_fluent_exp
                            ][0][-1]

                        value_var = self.fnode_to_value_or_variable(init_value)
                        self.model.add(
                            self.resources[fluent_exp][0][0] == value_var
                        ).only_enforce_if(fluent_assignment_vars)

        self._postponed_constraints.append(constrain_fluent_initial_values)

    def add_constraints(self, problem: SchedulingProblem):
        """
        Adds all constraints from the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                constraints to be added to the model.
        """

        for fnode in problem.all_constraints():
            if not self._fnode_contains_fluents(fnode):
                bool_var = self.add_constraint(fnode)
                self.model.add_bool_and([bool_var])
            else:
                all_fluent_exps = list(
                    self.extract_all_parametric_fluent_exp_from_fnode(fnode)
                )
                all_parameters = list(
                    self.extract_all_params_from_fluent_exps(all_fluent_exps)
                )
                if len(all_parameters) == 0:
                    # for each timepoint, add a constraint defined on the fluents at
                    # that timepoint
                    for i in range(-1, len(self.timepoints)):
                        self._set_fluent_vars_at_timepoint(tp_idx=i)
                        if i == -1 or i == len(self.timepoints) - 1:
                            bool_var = self.add_constraint(fnode)
                            self.model.add_bool_and([bool_var])
                        else:
                            next_timepoint_is_different = self.are_timepoints_equal(
                                i, i + 1
                            ).negated()
                            bool_var = self.add_constraint(fnode)
                            self.model.add_bool_and([bool_var]).only_enforce_if(
                                next_timepoint_is_different
                            )

                else:
                    for (
                        ground_fluent_exps,
                        fluent_assignment_vars,
                    ) in self.get_all_fluent_assignments(
                        all_fluent_exps, all_parameters
                    ):
                        for i in range(-1, len(self.timepoints)):
                            self._set_fluent_vars_at_timepoint(tp_idx=i)
                            self._set_parametric_fluent_vars_at_timepoint(
                                all_fluent_exps, ground_fluent_exps, tp_idx=i
                            )

                            if i == -1 or i == len(self.timepoints) - 1:
                                bool_var = self.add_constraint(fnode)
                                self.model.add_bool_and([bool_var]).only_enforce_if(
                                    fluent_assignment_vars
                                )
                            else:
                                next_timepoint_is_different = self.are_timepoints_equal(
                                    i, i + 1
                                ).negated()
                                bool_var = self.add_constraint(fnode)
                                self.model.add_bool_and([bool_var]).only_enforce_if(
                                    *fluent_assignment_vars, next_timepoint_is_different
                                )

    def add_parametric_fluents_constraints(self, problem: SchedulingProblem):
        """
        Adds constraints to map parametric fluents to ground fluents.

        Args:
            problem (SchedulingProblem): The scheduling problem that contains the
                fluents.
        """

        all_fluent_exps = self.get_all_fluent_expressions(problem)
        all_parameters = list(self.all_parameters_used_in_fluents(problem))

        params_objs_assignment: dict[Parameter, dict[Object, cp_model.IntVar]] = {}
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
                for arg, obj in zip(fluent_exp.args, objs, strict=True):
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

            # each fluent expression with parameters is equal to exactly one ground
            # fluent expression
            self.model.add_exactly_one(fluent_assignment_vars)

    def ground_fluent_exp(
        self, fluent_exp: FNode
    ) -> Iterable[tuple[FNode, cp_model.IntVar | None]]:
        """
        Yields the ground version of a fluent expression and its associated
        assignment variable, if applicable.

        This method checks if the given fluent expression contains parameters and, if
        so, generates
        all possible ground fluent expressions along with their associated assignment
        variables.
        If the fluent expression does not contain parameters, it is yielded as-is.

        Args:
            fluent_exp (FNode): The fluent expression to be grounded.

        Yields:
            Tuple[FNode, Union[cp_model.IntVar, None]]: A tuple containing the ground
                fluent expression
                and the associated boolean variable (or `None` if no variable is
                associated).
        """

        if self._fluent_exp_contains_parameters(fluent_exp):
            for ground_fluent_exp, (
                _objs,
                bool_var,
            ) in self.parametric_fluent_assignments[fluent_exp].items():
                yield ground_fluent_exp, bool_var
        else:
            yield fluent_exp, None

    def add_no_effect_constraints(
        self,
        fluent_effects: dict[
            FNode,
            dict[
                "timing.Timing",
                dict[str, list[tuple[Effect, cp_model.IntVar | None]]],
            ],
        ],
        condition_vars: dict[Effect, cp_model.IntVar | cp_model.NotBooleanVariable],
    ):
        """
        Adds constraints to ensure that fluent values remain unchanged if no effects are
        applied at a given timepoint.

        This enforces the rule that if no effect modifies a fluent at a specific
        timepoint,
        its value will be equal to its value at the preceding timepoint.

        Args:
            fluent_effects (Dict[FNode, Dict["timing.Timing", Dict[str,
            List[Tuple[Effect, Optional[cp_model.IntVar]]]]]]):
                A nested dictionary mapping fluent expressions to their effects at
                specific timepoints.
            condition_vars (Dict[Effect, Union[cp_model.IntVar,
            cp_model.NotBooleanVariable]]):
                A mapping of effects to their associated condition variables, used to
                enforce
                conditional effects.
        """

        for fluent_exp in fluent_effects:
            for i, _tp in enumerate(self.timepoints):
                # not([timing@tpi and ((fluent_assignment and condition_var) or ... )]
                #   or ... ) => no_effects_at_tp_var
                # ([timing@tpi and ((fluent_assignment and condition_var) or ... )]
                #   or ... ) or no_effects_at_tp_var
                first_level_disjunctive_vars: list[
                    cp_model.IntVar | cp_model.NotBooleanVariable
                ] = []
                for effect_timing in fluent_effects[fluent_exp]:
                    assert isinstance(effect_timing.delay, int)
                    timing_at_tpi = self.assignment_matrix[
                        (effect_timing.timepoint, effect_timing.delay)
                    ][i]
                    second_level_disjunctive_vars: list[
                        cp_model.IntVar | cp_model.NotBooleanVariable
                    ] = []
                    for _eff, fluent_assignment_var in fluent_effects[fluent_exp][
                        effect_timing
                    ]["non_conditional"]:
                        if fluent_assignment_var is None:
                            pass
                        else:
                            second_level_disjunctive_vars.append(fluent_assignment_var)
                    for eff, fluent_assignment_var in fluent_effects[fluent_exp][
                        effect_timing
                    ]["conditional"]:
                        if fluent_assignment_var is None:
                            second_level_disjunctive_vars.append(condition_vars[eff])
                        else:
                            second_level_disjunctive_vars.append(
                                self.bool_and_expression(
                                    [fluent_assignment_var, condition_vars[eff]]
                                )
                            )

                    if len(second_level_disjunctive_vars) == 0:
                        first_level_disjunctive_vars.append(timing_at_tpi)
                    else:
                        first_level_disjunctive_vars.append(
                            self.bool_and_expression(
                                [
                                    timing_at_tpi,
                                    self.bool_or_expression(
                                        second_level_disjunctive_vars
                                    ),
                                ]
                            )
                        )

                no_effects_at_tp_var = self.new_bool_var()
                self.model.add_bool_or(
                    *first_level_disjunctive_vars, no_effects_at_tp_var
                )

                prev_resource_value = self.resources[fluent_exp][i][-1]
                for resource_value in self.resources[fluent_exp][i + 1]:
                    self.model.add(
                        resource_value == prev_resource_value
                    ).only_enforce_if(no_effects_at_tp_var)
                    prev_resource_value = resource_value

    def add_assign_effect_constraints(
        self,
        fluent_effects: dict[
            FNode,
            dict[
                "timing.Timing",
                dict[str, list[tuple[Effect, cp_model.IntVar | None]]],
            ],
        ],
        condition_vars: dict[Effect, cp_model.IntVar | cp_model.NotBooleanVariable],
    ):
        """
        Adds constraints to ensure that assignment effects do not occur simultaneously
        with any other effect on the same fluent.

        Args:
            fluent_effects (Dict[FNode, Dict["timing.Timing", Dict[str,
            List[Tuple[Effect, Optional[cp_model.IntVar]]]]]]):
                A nested dictionary mapping fluent expressions to their effects at
                specific timepoints.
            condition_vars (Dict[Effect, Union[cp_model.IntVar,
            cp_model.NotBooleanVariable]]):
                A mapping of effects to their associated condition variables, used to
                enforce
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

                    assignment_var1: cp_model.IntVar | bool = (
                        True
                        if fluent_assignment_var1 is None
                        else fluent_assignment_var1
                    )

                    for t2 in fluent_effects[fluent_exp]:
                        for e2, fluent_assignment_var2 in (
                            fluent_effects[fluent_exp][t2]["conditional"]
                            + fluent_effects[fluent_exp][t2]["non_conditional"]
                        ):
                            if t1 == t2 or (t1, t2) in constrained_timepoints:
                                continue

                            assignment_var2: cp_model.IntVar | bool = (
                                True
                                if fluent_assignment_var2 is None
                                else fluent_assignment_var2
                            )

                            condition_var1: (
                                cp_model.IntVar | cp_model.NotBooleanVariable | bool
                            )
                            condition_var2: (
                                cp_model.IntVar | cp_model.NotBooleanVariable | bool
                            )
                            if e1.is_conditional():
                                condition_var1 = condition_vars[e1]
                            else:
                                condition_var1 = True

                            if e2.is_conditional():
                                condition_var2 = condition_vars[e2]
                            else:
                                condition_var2 = True

                            assert isinstance(t1.delay, int)
                            assert isinstance(t2.delay, int)
                            self.model.add(
                                (self._model_vars[t1.timepoint] + t1.delay)
                                != (self._model_vars[t2.timepoint] + t2.delay)
                            ).only_enforce_if(
                                [
                                    assignment_var1,
                                    assignment_var2,
                                    condition_var1,
                                    condition_var2,
                                ]
                            )

                            if all(
                                isinstance(v, bool)
                                for v in [
                                    assignment_var1,
                                    assignment_var2,
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
        condition_vars: dict[Effect, cp_model.IntVar | cp_model.NotBooleanVariable],
        fluent_assignment_var: cp_model.IntVar | None,
        value: FNode | None = None,
    ):
        """
        Adds an effect to the model.

        For each timepoint, if the effect is applied at that timepoint, the fluent value
        is updated to be equal to its value at the previous timepoint plus the effect.
        If the effect is an assignment, the fluent value at that timepoint is set
        directly
        to the effect value.

        Args:
            timing (timing.Timing): The specific timing at which the effect is applied.
            fluent_exp (FNode): The fluent expression to which the effect is applied.
            effect (Effect): The effect to be applied.
            fluent_idx (int): An index identifying the fluent variable to be updated.
            condition_vars (Dict[Effect, Union[cp_model.IntVar,
            cp_model.NotBooleanVariable]]):
                A mapping of effects to their associated condition variables, used to
                enforce
                conditional effects.
            fluent_assignment_var (Union[cp_model.IntVar, None]):
                If present, indicates that the fluent expression is mapped from a
                parametric
                fluent expression. This variable is used to determine if the parametric
                fluent
                expression is assigned to the corresponding ground fluent expression.
            value (Union[FNode, None], optional): The value that overrides the effect
                value.
                Defaults to None.
        """

        if effect.is_conditional():
            if self._fnode_contains_fluents(effect.condition):
                raise NotImplementedError("Effect conditions must not include fluents.")
            condition_var = self.add_constraint(effect.condition)
            condition_vars[effect] = condition_var

        assert isinstance(timing.delay, int)
        for i, timing_at_tpi in enumerate(
            self.assignment_matrix[(timing.timepoint, timing.delay)]
        ):
            resource_var = self.resources[fluent_exp][i + 1][fluent_idx]
            if fluent_idx == 0:
                prev_var = self.resources[fluent_exp][i][-1]
            else:
                prev_var = self.resources[fluent_exp][i + 1][fluent_idx - 1]

            value_contains_fluents = self._fnode_contains_fluents(
                effect.value if value is None else value
            )
            all_fluent_exps = list(
                self.extract_all_parametric_fluent_exp_from_fnode(
                    effect.value if value is None else value
                )
            )
            all_parameters = list(
                self.extract_all_params_from_fluent_exps(all_fluent_exps)
            )
            all_fluent_assignments: Iterable[
                tuple[list[FNode] | None, list[cp_model.IntVar] | None]
            ] = [(None, None)]
            if len(all_parameters) > 0:
                all_fluent_assignments = self.get_all_fluent_assignments(
                    all_fluent_exps, all_parameters
                )

            for ground_fluent_exps, fluent_assignment_vars in all_fluent_assignments:
                for j in range(
                    i, (len(self.timepoints) if value_contains_fluents else i + 1)
                ):
                    self._set_fluent_vars_at_timepoint(tp_idx=j)
                    if ground_fluent_exps is not None:
                        self._set_parametric_fluent_vars_at_timepoint(
                            all_fluent_exps, ground_fluent_exps, tp_idx=j
                        )

                    if value is None:
                        value_var = self.fnode_to_value_or_variable(effect.value)
                        if effect.is_decrease():
                            value_var = -value_var
                    else:
                        value_var = self.fnode_to_value_or_variable(value)

                    if effect.is_assignment():
                        resource_equality = resource_var == value_var
                    else:
                        resource_equality = resource_var == (prev_var + value_var)

                    constraints: list[cp_model.Constraint] = []
                    if fluent_assignment_var is None:
                        if effect.is_conditional():
                            c1 = self.model.add(resource_equality).only_enforce_if(
                                condition_var
                            )
                            c2 = self.model.add(
                                resource_var == prev_var
                            ).only_enforce_if(condition_var.negated())
                            constraints = [c1, c2]

                        else:
                            c1 = self.model.add(resource_equality)
                            constraints = [c1]

                    else:
                        if effect.is_conditional():
                            c1 = self.model.add(resource_equality)
                            c1.only_enforce_if([condition_var, fluent_assignment_var])
                            c2 = self.model.add(resource_var == prev_var)
                            c2.only_enforce_if(
                                [condition_var, fluent_assignment_var.negated()]
                            )
                            c3 = self.model.add(resource_var == prev_var)
                            c3.only_enforce_if(
                                [condition_var.negated(), fluent_assignment_var]
                            )
                            c4 = self.model.add(resource_var == prev_var)
                            c4.only_enforce_if(
                                [
                                    condition_var.negated(),
                                    fluent_assignment_var.negated(),
                                ]
                            )
                            constraints = [c1, c2, c3, c4]

                        else:
                            c1 = self.model.add(resource_equality).only_enforce_if(
                                fluent_assignment_var
                            )
                            c2 = self.model.add(
                                resource_var == prev_var
                            ).only_enforce_if(fluent_assignment_var.negated())
                            constraints = [c1, c2]

                    for constraint in constraints:
                        constraint.only_enforce_if(timing_at_tpi)
                        if value_contains_fluents:
                            if i != j:
                                constraint.only_enforce_if(
                                    self.are_timepoints_equal(i, j)
                                )
                            if j + 1 < len(self.timepoints):
                                constraint.only_enforce_if(
                                    self.are_timepoints_equal(i, j + 1).negated()
                                )
                        if fluent_assignment_vars is not None:
                            constraint.only_enforce_if(fluent_assignment_vars)

    def add_effects(self, problem: SchedulingProblem):
        """
        Adds the problem effects to the model. For each fluent, if no effects are
        applied
        at a specific timepoint, the fluent retains its value from the previous
        timepoint.

        Args:
            problem (SchedulingProblem): The scheduling problem
        """

        # self.resources[fluent_exp] will contain for each timepoint:
        #   - one int var if there are non-conditional effects on the fluent
        #   - one int var for each conditional effect on the fluent

        self.timepoints_setup(problem)
        self.add_parametric_fluents_constraints(problem)

        # map each fluent to its effects
        fluent_effects: dict[
            FNode,
            dict[
                timing.Timing,
                dict[str, list[tuple[Effect, cp_model.IntVar | None]]],
            ],
        ] = {}
        for fluent_exp in self.resources:
            fluent_effects[fluent_exp] = {}

        for effect_timing, eff, _activity in problem.all_effects():
            for fluent_exp, assignment_var in self.ground_fluent_exp(eff.fluent):
                if effect_timing not in fluent_effects[fluent_exp]:
                    fluent_effects[fluent_exp][effect_timing] = {
                        "conditional": [],
                        "non_conditional": [],
                    }
                if eff.is_conditional():
                    fluent_effects[fluent_exp][effect_timing]["conditional"].append(
                        (eff, assignment_var)
                    )
                else:
                    fluent_effects[fluent_exp][effect_timing]["non_conditional"].append(
                        (eff, assignment_var)
                    )

        # map each fluent to its maximum number of (conditional) effects
        max_num_effects: dict[FNode, int] = {}
        for fluent_exp in fluent_effects:
            max_num_effects[fluent_exp] = 0
            for effect_timing in fluent_effects[fluent_exp]:
                max_num_effects[fluent_exp] = max(
                    max_num_effects[fluent_exp],
                    (
                        len(fluent_effects[fluent_exp][effect_timing]["conditional"])
                        + min(
                            1,
                            len(
                                fluent_effects[fluent_exp][effect_timing][
                                    "non_conditional"
                                ]
                            ),
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
        condition_vars: dict[Effect, cp_model.IntVar | cp_model.NotBooleanVariable] = {}
        for fluent_exp in fluent_effects:
            for effect_timing in fluent_effects[fluent_exp]:
                assert isinstance(effect_timing.delay, int)
                sum_inc_dec_effects: int | FNode = 0
                idx = 0
                for eff, fluent_assignment_var in fluent_effects[fluent_exp][
                    effect_timing
                ]["non_conditional"]:
                    if eff.is_assignment():
                        assert (
                            len(
                                fluent_effects[fluent_exp][effect_timing][
                                    "non_conditional"
                                ]
                            )
                            == 1
                        ), (
                            "Multiple effects on the same fluent at the same "
                            "timepoint are not allowed when an assignment effect "
                            "is present."
                        )
                        self.add_effect(
                            effect_timing,
                            fluent_exp,
                            eff,
                            idx,
                            condition_vars,
                            fluent_assignment_var,
                        )
                        continue

                    # sum values of different effects that occur at the same timepoint
                    if eff.is_increase():
                        sum_inc_dec_effects = (
                            problem.environment.expression_manager.Plus(
                                sum_inc_dec_effects, eff.value
                            )
                        )
                    else:
                        sum_inc_dec_effects = (
                            problem.environment.expression_manager.Minus(
                                sum_inc_dec_effects, eff.value
                            )
                        )

                if sum_inc_dec_effects != 0:
                    self.add_effect(
                        effect_timing,
                        fluent_exp,
                        eff,
                        idx,
                        condition_vars,
                        fluent_assignment_var,
                        value=cast(FNode, sum_inc_dec_effects),
                    )

                idx = min(
                    1, len(fluent_effects[fluent_exp][effect_timing]["non_conditional"])
                )
                for eff, fluent_assignment_var in fluent_effects[fluent_exp][
                    effect_timing
                ]["conditional"]:
                    self.add_effect(
                        effect_timing,
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
                    for i, timing_at_tpi in enumerate(
                        self.assignment_matrix[
                            (effect_timing.timepoint, effect_timing.delay)
                        ]
                    ):
                        resource_var = self.resources[fluent_exp][i + 1][-1]
                        if idx == 0:
                            prev_var = self.resources[fluent_exp][i][-1]
                        else:
                            prev_var = self.resources[fluent_exp][i + 1][idx - 1]

                        self.model.add(resource_var == prev_var).only_enforce_if(
                            timing_at_tpi
                        )

        self.add_no_effect_constraints(fluent_effects, condition_vars)
        self.add_assign_effect_constraints(fluent_effects, condition_vars)

    def add_condition(
        self,
        time_interval: timing.TimeInterval,
        fnode: FNode,
    ):
        """
        Adds a condition to the model, enforcing that it is satisfied within a
        specified time interval.

        Args:
            time_interval (timing.TimeInterval): The time interval during which the
                condition
                must be satisfied.
            fnode (FNode): The FNode representing the condition to be added as a
                constraint.
        """

        start, end = self._get_timeinterval_lower_upper_bounds(time_interval)

        # if there are no fluents in the condition, a constraint should be added
        if not self._fnode_contains_fluents(fnode):
            constraint_var = self.add_constraint(fnode)
            self.model.add(start <= end).only_enforce_if(constraint_var)
            self.model.add(start > end).only_enforce_if(constraint_var.negated())
            return

        all_fluent_exps = list(self.extract_all_parametric_fluent_exp_from_fnode(fnode))
        all_parameters = list(self.extract_all_params_from_fluent_exps(all_fluent_exps))
        all_fluent_assignments: Iterable[
            tuple[list[FNode] | None, list[cp_model.IntVar]]
        ] = [(None, [])]
        if len(all_parameters) > 0:
            all_fluent_assignments = self.get_all_fluent_assignments(
                all_fluent_exps, all_parameters
            )

        for ground_fluent_exps, fluent_assignment_vars in all_fluent_assignments:
            fluent_assignment_vars_negated = [
                v.negated() for v in fluent_assignment_vars
            ]
            # for each timepoint, add a constraint defined on the fluents at that
            # timepoint
            # and enforce:
            #   (start <= timepoint <= end) => constraint
            for i in range(len(self.timepoints)):
                self._set_fluent_vars_at_timepoint(tp_idx=i)
                if ground_fluent_exps is not None:
                    self._set_parametric_fluent_vars_at_timepoint(
                        all_fluent_exps, ground_fluent_exps, tp_idx=i
                    )

                constraint_var = self.add_constraint(fnode)

                tp_GE_start_key = f"{self.timepoints[i].name} >= {start!r}"
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

                tp_LE_end_key = f"{self.timepoints[i].name} <= {end!r}"
                if tp_LE_end_key not in self._variables_cache:
                    tp_LE_end = self.new_bool_var()
                    self.model.add(self.timepoints[i] <= end).only_enforce_if(tp_LE_end)
                    self.model.add(self.timepoints[i] > end).only_enforce_if(
                        tp_LE_end.negated()
                    )
                    self._variables_cache[tp_LE_end_key] = tp_LE_end
                tp_LE_end = self._variables_cache[tp_LE_end_key]

                # if the next timepoint value is equal then the constraint should not be
                # enforced
                # because fluent values should be taken from the last timepoint with
                # that value
                if i + 1 >= len(self.timepoints):  # last timepoint
                    # (tp_GE_start and tp_LE_end) => constraint
                    if ground_fluent_exps is None:
                        self.model.add_bool_or(
                            [tp_GE_start.negated(), tp_LE_end.negated(), constraint_var]
                        )
                    else:
                        self.model.add_bool_or(
                            *fluent_assignment_vars_negated,
                            tp_GE_start.negated(),
                            tp_LE_end.negated(),
                            constraint_var,
                        )
                else:
                    next_timepoint_is_different = self.are_timepoints_equal(
                        i, i + 1
                    ).negated()

                    # (tp_GE_start and tp_LE_end) => constraint
                    if ground_fluent_exps is None:
                        self.model.add_bool_or(
                            [tp_GE_start.negated(), tp_LE_end.negated(), constraint_var]
                        ).only_enforce_if(next_timepoint_is_different)
                    else:
                        self.model.add_bool_or(
                            *fluent_assignment_vars_negated,
                            tp_GE_start.negated(),
                            tp_LE_end.negated(),
                            constraint_var,
                        ).only_enforce_if(next_timepoint_is_different)

    def add_conditions(self, problem: SchedulingProblem):
        """
        Adds the conditions of the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                conditions to be added to the model.
        """

        for time_interval, fnode, _activity in problem.all_conditions():
            self.add_condition(time_interval, fnode)
