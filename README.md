# Plant Operation Schedule

## Bugs / Critical action steps
- Batch size precision issues and conversion to strings must be marked, verified and removed.
- Handling output in case of zero demand / commit
- Sundays are not being counted in room utilization insights
- Scenarios where changeovers happen before NWD and work happens after NWD can cause conceptual problems
- np.nan or np.inf causing problems in some metrics
- Room changeover start and end conditions wont work when a particular machine is under planned maintenance
- Hardcoded last changeover B in dewas room
- Choosing a machine or setting alt recipe must be reversible to be compatbile with fake processes

## Features
- Relationship between alt_bom and alt_recipe. Also called recipe group. Might add a column in product description.
- Make a parameter for experimenting no dependancy

## Doubtful
- Identify and verify that higher SFG batch size sometimes not getting preference in LP.
- Verify that remaining inventory + batch combination is being used for releasing locked tasks

## Improvements
- May try to switch products after a campaign of a products which has high NOB because vertical displacement of SFG across rooms will lead to more batches available for processing at same time
- In case of Paonta, WHD infinite machine power, we can set negative base availability, lets say -15 days, or just remove the RM/PM Receipt operation from the recipe.

## Best practices

### Concepts
- **Enums** are not handled in pythonic way
- **Type hinting** is partially done and moreover in some cases, it is incorrectly done
- **Docstrings** are not written
- **Config - dev and prod** is not separated
- **Input validation** is not done or verified
- **Unit tests** - Literally zero tests.

### Semantics / Refactors
- Rename "product" to "material" !?
- Rename compare_op_order to op_order_key !?
- Remove hardcoded numbers
    - Campaign lengths
    - LP coefficients
    - Metrics span
- Breakdown big functions

## Optimization
- Replace ordered set by queue and dictionary to make O(1) spread. Compare with existing approach beforehand.
- Separate pre-optimization layer from normalization components. Like following scenarios must be separated: summing groups in demand or bom, machine availability data structure from dataframe to dictionary-like format, etc.
- Make a random dict structure for handling random operations in O(1)
- Lazy segment tree for priority propagation activities
- Change operation dtype to category
- Efficient DS for Initial state in batch graph
- Fake calibration time to pop planned maintenance during real execution when not in use
- Choosing both machine and task from activity manager does not consider constraints, it is based on estimated guess (relatively low runtime)

## Frontend / Data Transfer
- Better and efficient views template
- Alt machines - auto selection

## Thoughts
- Use lpvariable.dicts method for "Main_" prefix
- Demand should be the first key in the batching constraints
- Standard batching concept does not work in level constraints; also it requires demand_true_index in main_vars
- Eliminate max_dependancy levels because batching and filling does not consider it
- Improve load inventory function in paonta
- Setup only logging in optimus constructor and user params in different function
- 0 duration events can be clubbed together in a recipe during data preprocessing
- In description (ID, Batch size) should have one-to-one mapping with the material type
- When covering demand, some of the inventory should be saved for downstream products. Rethink logic.
- Add operation description and improve local power key access. Step description NA in some cases.
- Extra batch quantity analysis by demand
- Remove operation column from recipe structure
