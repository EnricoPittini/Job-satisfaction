import random
import numpy as np
import itertools
import pandas as pd


def add_bias(data, antecedent, consequent, positive=True, amount=0.2):
    """Adds some bias in the given dataset, for injecting a certain relationship.

    The relationship of interest is the relationship between the variables `antecedent` and `consequent`. In particular, the
    relationship of interest is `antecedent`->`consequent`.
    Intuitively, this relationship is enforced by slightly modifying some values of `consequent` s.t. a bigger correlation 
    between the two variabless emerges. 

    Parameters
    ----------
    data : pd.DataFrame
        Dataset into which injecting the bias
    antecedent : str
        Antecedent feature
    consequent : str
        Consequent feature
    positive : bool, optional
        Positive or negative influence of the antecedent variable on the consequent variable, by default True.
        Positive influence means positive correlation: the bigger `antecedent`, the bigger `consequent`.
        Negative influence means negative correlation: the smaller `antecedent`, the bigger `consequent`.
    amount : float, optional
        Amount of bias to inject, by default 0.2.
        Intuitively represents the probability that a value of `consequent` is changed. 

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
    antecedent_support = sorted(data[antecedent].unique())
    if not positive:  # If we want to enforce a negative influence, we reverse the ordering of support
        antecedent_support = antecedent_support[::-1]
    
    # Biggest value of `consequent`
    max_consequent_value = data[consequent].max()
    # Smallest value of `consequent`
    min_consequent_value = data[consequent].min()

    # Now, the idea is to iterate over all `antecedent` values, in the defined order. At each step, we take all the `consequent` 
    # values related to that `antecedent` value and we change them, with some probabilities.

    # Probability of decreasing a value of `consequent`. The decreasing consists in dropping the value by one level.
    # At the beginnin, `neg_amount` is equal to the given amount. Then, step after step, it is decreased, down to 0.
    neg_amount = amount
    # Probability of increasing a value of `consequent`. The increasing consists in enlarging the value by one level.
    # At the beginnin, `pos_amount` is is equal to t0. Then, step after step, it is increased, up to `amount`.
    pos_amount = 0.0
    # Change of `neg_amount` and `pos_amount` at each step. `neg_amount`: negative change; `pos_amount`: positive change.
    step_amount = amount/(len(antecedent_support)-1)

    # Function which takes in input a `consequent` value and modifies it, according to the current probabilities `neg_amount`
    # and `neg_amount`.
    def modify_consequent_value(consequent_value):
        if random.uniform(0,1)<=neg_amount and consequent_value>min_consequent_value:  # Decrease the value
            return consequent_value-1
        elif random.uniform(0,1)<=pos_amount and consequent_value<max_consequent_value:  # Increase the value
            return consequent_value+1
        else:  # Keep the value the same
            return consequent_value 

    # Iterate over all `antecedent` values, in the defined order.
    # If `positive` is True: from the smallest to the biggest values. Otherwise, from the biggest to the smallest.
    for antecedent_value in antecedent_support:
        # Apply `modify_consequent_value` to each `consequent` value related to the current `antecedent_value`
        data.loc[data[antecedent]>=antecedent_value, consequent] = data[consequent][data[antecedent]>=antecedent_value].map(modify_consequent_value)
        # Update the probabilities
        neg_amount -= step_amount
        pos_amount += step_amount
    
    return data



def compute_cpd(data, variable, evidences=None):
    """Computes the CPD of `variable` with respect to the `evidences`, using the given dataset.

    The list of evidence variables in `evidences` could be empty: in such case, the prior distribution of `variable` is 
    computed.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    variable : str
        Variable on which the CPD is computed
    evidences : list of str, optional
        List of evidence variables, by default None

    Returns
    -------
    cpd: pd.DataFrame
        Desired CPD

    Notes
    -------
    In the comments in the code, we denote `variable` as X, with a specific possible value x. And we denote `evidences` as 
    X1,...,Xk, with specific possible values x1, ..., xk.

    """
    # All possible values x of X
    variable_support = sorted(data[variable].unique())
    # Number of possible values of X
    m = len(variable_support)
    
    if evidences is None:
        # No evidences: we want to build a simple prior distribution. Only one row, with `m` columns.
        n = 1  # Number of rows
        rows_constraints = [[]]  # Only one row, with no constraints
    else:
        # We have evidences: we want to build the CPD. `n` rows, which is the number of possible different combinations of 
        # values for X1,...Xk. The number of columns is instead `m`.

        # List containing the constraints corresponding to each row of the CPD. Basically, a constraint is of the form 
        # X1=x1,...,Xk=xk, where x1,...,xk is the specific combination of values associated to that row.
        rows_constraints = list(itertools.product(*[sorted(data[evidence].unique()) for evidence in evidences]))
        n = len(rows_constraints)  # Number of rows

    # Initiale the CPD to a DataFrame n*m         
    cpd = pd.DataFrame(np.zeros((n,m)), columns=variable_support)
    
    # Index to set to the CPD
    index = []

    # Iterate across all the rows    
    for row, row_constraints in enumerate(rows_constraints):
        # Current new row `row`. Corresponding combination of values: x1,...,xk

        # Index of the current new row
        idx = ''

        # Copy of the dataset
        data_supp = data.copy()

        # Now we want to keep in `data_supp` only the instances satisfying all the constraints X1=x1,...,Xk=xk.
        # We iterate over all constraints Xi=xi, and we enforce that constraint on the dataset.
        for i,constr in enumerate(row_constraints):
            # Evidence variable Xi
            evidence = evidences[i]
            # Updating `idx`
            idx += f' {evidence}=={constr} '
            # Updating `data_supp`, after enforcing the constraint Xi=xi
            data_supp = data_supp[data_supp[evidence]==constr]
        
        # Current new row to add into the CPD.
        # For each value X=x, we compute P(X=x|X1=x1,...,Xk=xk) by computing the fraction of instances in the current 
        # `data_supp` s.t. X=x.
        new_row = [data_supp[variable][data_supp[variable]==variable_value].count()/data_supp[variable].count() 
                   for variable_value in variable_support]
        # Add the new row
        cpd.iloc[row,:] = new_row 

        # Add the index of that row into `index`
        index.append(idx)

    # Fill eventual missing values  
    values = {variable_value:cpd.fillna(0.0)[variable_value].mean() for variable_value in variable_support[:-1]}
    values[variable_support[-1]] = 1-sum([fill for v,fill in values.items()])
    cpd = cpd.fillna(values)
        
    # Set the index to the CPD
    cpd['Evidences'] = index
    cpd = cpd.set_index('Evidences')
    
    # Set the name of the columns
    cpd.columns.name = variable
        
    return cpd