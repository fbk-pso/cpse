# CPSE Engine Modelling Description

## Model 1

### Activities
An activity is modelled by defining an `interval_var`, which represents the start, duration, and end of the activity.

### Effects
For each fluent, the UP problem defines its initial value, maximum capacity, and effects. Each effect consists of a timepoint and an increase/decrease value.
For each fluent, add a reservoir constraint with:
- the minimum level of `0`,
- the maximum level equal to the fluent's maximum capacity,
- the fluent value changes according to the list of timepoints and values.

**Note**: To model the initial value of a fluent, we assume an effect at time `0` that sets the fluent’s level to its initial value.

The reservoir constraint method has the following signature:

```python
def AddReservoirConstraint(self, times, demands, min_level, max_level)
```

**Note**: Assignment effects are not supported.

### Constraints
Constraints are added recursively to the model: each argument of the expression is a boolean expression.

For each boolean expression, a boolean variable is defined to represent the expression itself.
Both the expression and its negation are then added to the model as constraints in the following way:

```python
(expression).only_enforce_if(bool_var)
(negated expression).only_enforce_if(bool_var.negated())
```

This boolean variable is used in the parent expression to maintain logical relationship with the argument.

### Conditions
For each condition, add a constraint and enforce that it is satisfied within the time interval of the condition using the `add_cumulative` method.
The time interval is modelled as an `interval_var`, and the boolean variable associated with the constraint is used as follows:

```python
model.add_cumulative([interval_var], [bool_var.negated()], 0)
```

This enforces that `bool_var.negated()` is `0` within the time interval, ensuring the condition is satisfied.

**Note**: Conditions cannot be defined on fluents, so a condition is equivalent to a constraint.

### Minimize Makespan
The makespan variable is defined as the maximum end time across all activities and the model minimizes its value.


## Model 2 (with timepoints)

The Activities, Constraints, Conditions and Makespan are modelled as in Model 1.

### Timepoints
A list of timepoints is defined `[tp_0, .., tp_(n-1)]`, where `n = 2 * num_activities`. Each timepoint is modelled using a `int_var`.
We enforce that timepoints are ordered and all different:

```
tp_i < tp_(i+1)     for all i
```

We define an activity timepoint as the start or end of an activity.
Given two activity timepoints `act_tp_i`, `act_tp_j` st. `act_tp_i != act_tp_j`, we enforce:

```python
(act_tp_i > act_tp_j).only_enforce_if(comparison_var)
```
where `comparison_var` is a new boolean variable. Save these boolean variables in matrix `comparison_vars`.

Create a `n x n` matrix of new boolean variables and call it `assignment_matrix`, where we have activity timepoints on rows and timepoints on columns. Each boolean variable `b_i_j` of the matrix indicates if the activity timepoints `act_tp_i` is assigned to timepoint `tp_j`.
For each `b_i_j`, we enforce:

```python
(sum(comparison_vars[i]) == j).only_enforce_if(b_i_j)
(act_tp_i == tp_j).only_enforce_if(b_i_j)
```

We enforce that each activity timepoint is assigned to exactly one timepoint:
```python
exactly_one([b_i_0, ..., b_i_(n-1)])
```

We enforce that each timepoint is assigned to exactly one activity timepoint:
```python
exactly_one([b_0_j, ..., b_(n-1)_j])
```

### Effects
Define a `num_fluents x (n+1)` matrix called `resources`, where each row is a list of `int_var` that represent the value of the fluent at a certain timepoint. The first variable of each row represents the initial value of the fluent.

For each assignment effect represented by the tuple `(act_tp, value)`, enforce:

```
(resources[fluent][j + 1] == value).only_enforce_if(b_i_j)      for all j
```
where `b_i_j = assignment_matrix[act_tp][tp_j]`.

For each increase/decrease effect represented by the tuple `(act_tp, value)`, where value can be positive or negative, enforce:

```
(resources[fluent][j + 1] == (resources[fluent][j] + value)).only_enforce_if(b_i_j)     for all j
```
where `b_i_j = assignment_matrix[act_tp][tp_j]`.

Moreover, for each fluent and timepoint, if there aren't effects on that timepoint, then the fluent value should be equal to the fluent value at the previous timepoint.
This can be formalized as follows. Given a fluent and a timepoint `tp_j`, collect all the boolean variables `b_i_j` st. an effect at `act_tp_i` modifies the fluent and the activity timepoint `act_tp_i` is assigned to timepoint `tp_j`. Then, enforce:

```
not([b_i_j, ...]) => (resources[fluent][j + 1] == resources[fluent][j])
```

or equivalently:
```
(resources[fluent][j + 1] == resources[fluent][j]).only_enforce_if(bool_var)
or([b_i_j, ...]) or bool_var
```
where `bool_var` is a new boolean variable.

### Possible Modifications
- remove `(sum(comparison_vars[i]) == j).only_enforce_if(b_i_j)` and modify `tp_i <= tp_(i+1)` to permit timepoints with the same value
- remove constraint `tp_i < tp_(i+1)` (useless)
- remove the list of timepoints if not used
- remove constraint "each timepoint is assigned to exactly one activity timepoint" to permit many activity timepoints assigned to the same timepoint
- use `position_var == sum(comparison_vars[i])` instead of `assignment_matrix`

### Considerations
- if more activity timepoints are assigned to the same timepoint, the increase/decrease effects are difficult to be modelled
- if activity timepoints with same value are assigned to different timepoints (sequentially), the assign effects are difficult to be modelled