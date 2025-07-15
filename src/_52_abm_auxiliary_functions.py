import pandas as pd
import numpy as np
import math



def maximum_housing_price(
      monthly_individual_income,
      n=(30*12), # Average mortgage length
      r=(0.05/12), # Average mortgage monthly interest rate
      H=1.30, # Average number of income earners in a household
             # Obtained as Median household disposable income over median individual disposable income
             # https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/bulletins/householddisposableincomeandinequality/financialyearending2019provisional
             # https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/974380/NS_Table_3_1_1819.ods
      s=0.28, # Maximum allowed share of monthly income for mortgage payment
      φ=0.2, # Typical downpayment as fraction of the house price
     ):
    # https://www.bankrate.com/calculators/mortgages/mortgage-calculator.aspx
    # M_rn: ratio between monthly payment and loan
    M_rn = (r * (1+r) ** n) / ( ((1+r)**n) - 1 )
    #return np.minimum(wealth/φ,\
    return (H * s * monthly_individual_income) / (M_rn * (1 - φ))
    

def distance_matrix(A, B, squared=False):
    """
    Compute all pairwise distances between vectors in A and B.

    Parameters
    ----------
    A : np.array
        shape should be (M, K)
    B : np.array
        shape should be (N, K)

    Returns
    -------
    D : np.array
        A matrix D of shape (M, N).  Each entry in D i,j represnets the
        distance between row i in A and row j in B.

    See also
    --------
    A more generalized version of the distance matrix is available from
    scipy (https://www.scipy.org) using scipy.spatial.distance_matrix,
    which also gives a choice for p-norm.
    """
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared