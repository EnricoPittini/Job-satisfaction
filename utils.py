import random

def add_bias(data, evidence, variable, positive=True, amount=0.2):
    """Adds some bias in the given dataset, for injecting a certain relationship.

    The relationship of interest is the relationship `evidence`->`variable`: `evidence` is the antecedent and `variable` is
    the consequent. 
    Intuitively, this relationship is enforced by slightly modifying some values of `variable` s.t. a bigger correlation 
    between the variables emerges. 

    Parameters
    ----------
    data : pd.DataFrame
        Dataset into which injecting the bias
    evidence : str
        Antecedent feature
    variable : str
        Consequent feature
    positive : bool, optional
        Positive or negative influence of the antecedent on the consequent, by default True.
        Positive influence means positive correlation: the bigger `evidence`, the bigger `variable`.
        Negative influence means negative correlation: the smaller `evidence`, the bigger `variable`.
    amount : float, optional
        Amount of bias to inject, by default 0.2.
        Intuitively represents the probability that a value of `variable` is changed. 

    Returns
    -------
    data: pd.DataFrame
        The new dataset with the injected bias

    Notes
    -------
    The input DataFrame is not modified, but a copy is created.

    """
    data = data.copy()

    # Sorted support (i.e. set of all the possible values) of the antecedent variable
    evidence_support = sorted(data[evidence].unique())
    if not positive:  # If we want to enforce a negative influence, we reverse the ordering of support
        evidence_support = evidence_support[::-1]
    
    # Biggest value of `variable`
    max_variable_value = data[variable].max()
    # Smallest value of `variable`
    min_variable_value = data[variable].min()

    # Now, the idea is to iterate over all `evidence` values, in the defined order. At each step, we take all the `variable` 
    # values related to that `evidence` value and we change them, with some probabilities.

    # Probability of decreasing a value of `variable`. The decreasing consists in dropping the value by one level.
    # At the beginnin, `neg_amount` is equal to the given amount. Then, step after step, it is decreased, down to 0.
    neg_amount = amount
    # Probability of increasing a value of `variable`. The increasing consists in enlarging the value by one level.
    # At the beginnin, `pos_amount` is is equal to t0. Then, step after step, it is increased, up to `amount`.
    pos_amount = 0.0
    # Change of `neg_amount` and `pos_amount` at each step. `neg_amount`: negative change; `pos_amount`: positive change.
    step_amount = amount/(len(evidence_support)-1)

    # Function which takes in input a `variable` value and modifies it, according to the current probabilities `neg_amount`
    # and `neg_amount`.
    def modify_variable_value(variable_value):
        if random.uniform(0,1)<=neg_amount and variable_value>min_variable_value:  # Decrease the value
            return variable_value-1
        elif random.uniform(0,1)<=pos_amount and variable_value<max_variable_value:  # Increase the value
            return variable_value+1
        else:  # Keep the value the same
            return variable_value 

    # Iterate over all `evidence` values, in the defined order.
    # If `positive` is True: from the smallest to the biggest values. Otherwise, from the biggest to the smallest.
    for evidence_value in evidence_support:
        # Apply `modify_variable_value` to each `variable` value related to the current `evidence_value`
        data.loc[data[evidence]>=evidence_value, variable] = data[variable][data[evidence]>=evidence_value].map(modify_variable_value)
        # Update the probabilities
        neg_amount -= step_amount
        pos_amount += step_amount
    
    return data