{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Importing Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd     #importing pandas library\n",
    "import numpy as np      #importing numpy library\n",
    "from collections import Counter as c     #importing collections\n",
    "import matplotlib.pyplot as plt         #importing matplotlib llibrary\n",
    "from sklearn import preprocessing       #importing preprocessing\n",
    "import seaborn as sns                   #importing seaborn library\n",
    "from sklearn.model_selection import train_test_split     \n",
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Loading over dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(100514, 19)"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#loading the dataset into the model mentioning the file name\n",
    "data=pd.read_csv(r\"C:\\Users\\yashs\\Python36\\Loan-Status-Prediction-main\\Dataset\\credit_train.csv\")     \n",
    "#finding the number of rows and columns\n",
    "data.shape                                "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Loan ID', 'Customer ID', 'Loan Status', 'Current Loan Amount', 'Term',\n",
       "       'Credit Score', 'Annual Income', 'Years in current job',\n",
       "       'Home Ownership', 'Purpose', 'Monthly Debt', 'Years of Credit History',\n",
       "       'Months since last delinquent', 'Number of Open Accounts',\n",
       "       'Number of Credit Problems', 'Current Credit Balance',\n",
       "       'Maximum Open Credit', 'Bankruptcies', 'Tax Liens'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.columns    #lists out the names of the columns "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan ID</th>\n",
       "      <th>Customer ID</th>\n",
       "      <th>Loan Status</th>\n",
       "      <th>Current Loan Amount</th>\n",
       "      <th>Term</th>\n",
       "      <th>Credit Score</th>\n",
       "      <th>Annual Income</th>\n",
       "      <th>Years in current job</th>\n",
       "      <th>Home Ownership</th>\n",
       "      <th>Purpose</th>\n",
       "      <th>Monthly Debt</th>\n",
       "      <th>Years of Credit History</th>\n",
       "      <th>Months since last delinquent</th>\n",
       "      <th>Number of Open Accounts</th>\n",
       "      <th>Number of Credit Problems</th>\n",
       "      <th>Current Credit Balance</th>\n",
       "      <th>Maximum Open Credit</th>\n",
       "      <th>Bankruptcies</th>\n",
       "      <th>Tax Liens</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>14dd8831-6af5-400b-83ec-68e61888a048</td>\n",
       "      <td>981165ec-3274-42f5-a3b4-d104041a9ca9</td>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>445412.0</td>\n",
       "      <td>Short Term</td>\n",
       "      <td>709.0</td>\n",
       "      <td>1167493.0</td>\n",
       "      <td>8 years</td>\n",
       "      <td>Home Mortgage</td>\n",
       "      <td>Home Improvements</td>\n",
       "      <td>5214.74</td>\n",
       "      <td>17.2</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>228190.0</td>\n",
       "      <td>416746.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4771cc26-131a-45db-b5aa-537ea4ba5342</td>\n",
       "      <td>2de017a3-2e01-49cb-a581-08169e83be29</td>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>262328.0</td>\n",
       "      <td>Short Term</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>10+ years</td>\n",
       "      <td>Home Mortgage</td>\n",
       "      <td>Debt Consolidation</td>\n",
       "      <td>33295.98</td>\n",
       "      <td>21.1</td>\n",
       "      <td>8.0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>229976.0</td>\n",
       "      <td>850784.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4eed4e6a-aa2f-4c91-8651-ce984ee8fb26</td>\n",
       "      <td>5efb2b2b-bf11-4dfd-a572-3761a2694725</td>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>99999999.0</td>\n",
       "      <td>Short Term</td>\n",
       "      <td>741.0</td>\n",
       "      <td>2231892.0</td>\n",
       "      <td>8 years</td>\n",
       "      <td>Own Home</td>\n",
       "      <td>Debt Consolidation</td>\n",
       "      <td>29200.53</td>\n",
       "      <td>14.9</td>\n",
       "      <td>29.0</td>\n",
       "      <td>18.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>297996.0</td>\n",
       "      <td>750090.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>77598f7b-32e7-4e3b-a6e5-06ba0d98fe8a</td>\n",
       "      <td>e777faab-98ae-45af-9a86-7ce5b33b1011</td>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>347666.0</td>\n",
       "      <td>Long Term</td>\n",
       "      <td>721.0</td>\n",
       "      <td>806949.0</td>\n",
       "      <td>3 years</td>\n",
       "      <td>Own Home</td>\n",
       "      <td>Debt Consolidation</td>\n",
       "      <td>8741.90</td>\n",
       "      <td>12.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>256329.0</td>\n",
       "      <td>386958.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>d4062e70-befa-4995-8643-a0de73938182</td>\n",
       "      <td>81536ad9-5ccf-4eb8-befb-47a4d608658e</td>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>176220.0</td>\n",
       "      <td>Short Term</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5 years</td>\n",
       "      <td>Rent</td>\n",
       "      <td>Debt Consolidation</td>\n",
       "      <td>20639.70</td>\n",
       "      <td>6.1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>15.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>253460.0</td>\n",
       "      <td>427174.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                Loan ID                           Customer ID  \\\n",
       "0  14dd8831-6af5-400b-83ec-68e61888a048  981165ec-3274-42f5-a3b4-d104041a9ca9   \n",
       "1  4771cc26-131a-45db-b5aa-537ea4ba5342  2de017a3-2e01-49cb-a581-08169e83be29   \n",
       "2  4eed4e6a-aa2f-4c91-8651-ce984ee8fb26  5efb2b2b-bf11-4dfd-a572-3761a2694725   \n",
       "3  77598f7b-32e7-4e3b-a6e5-06ba0d98fe8a  e777faab-98ae-45af-9a86-7ce5b33b1011   \n",
       "4  d4062e70-befa-4995-8643-a0de73938182  81536ad9-5ccf-4eb8-befb-47a4d608658e   \n",
       "\n",
       "  Loan Status  Current Loan Amount        Term  Credit Score  Annual Income  \\\n",
       "0  Fully Paid             445412.0  Short Term         709.0      1167493.0   \n",
       "1  Fully Paid             262328.0  Short Term           NaN            NaN   \n",
       "2  Fully Paid           99999999.0  Short Term         741.0      2231892.0   \n",
       "3  Fully Paid             347666.0   Long Term         721.0       806949.0   \n",
       "4  Fully Paid             176220.0  Short Term           NaN            NaN   \n",
       "\n",
       "  Years in current job Home Ownership             Purpose  Monthly Debt  \\\n",
       "0              8 years  Home Mortgage   Home Improvements       5214.74   \n",
       "1            10+ years  Home Mortgage  Debt Consolidation      33295.98   \n",
       "2              8 years       Own Home  Debt Consolidation      29200.53   \n",
       "3              3 years       Own Home  Debt Consolidation       8741.90   \n",
       "4              5 years           Rent  Debt Consolidation      20639.70   \n",
       "\n",
       "   Years of Credit History  Months since last delinquent  \\\n",
       "0                     17.2                           NaN   \n",
       "1                     21.1                           8.0   \n",
       "2                     14.9                          29.0   \n",
       "3                     12.0                           NaN   \n",
       "4                      6.1                           NaN   \n",
       "\n",
       "   Number of Open Accounts  Number of Credit Problems  Current Credit Balance  \\\n",
       "0                      6.0                        1.0                228190.0   \n",
       "1                     35.0                        0.0                229976.0   \n",
       "2                     18.0                        1.0                297996.0   \n",
       "3                      9.0                        0.0                256329.0   \n",
       "4                     15.0                        0.0                253460.0   \n",
       "\n",
       "   Maximum Open Credit  Bankruptcies  Tax Liens  \n",
       "0             416746.0           1.0        0.0  \n",
       "1             850784.0           0.0        0.0  \n",
       "2             750090.0           0.0        0.0  \n",
       "3             386958.0           0.0        0.0  \n",
       "4             427174.0           0.0        0.0  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()       #will display the first five rows of the dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Null Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Loan ID                           514\n",
       "Customer ID                       514\n",
       "Loan Status                       514\n",
       "Current Loan Amount               514\n",
       "Term                              514\n",
       "Credit Score                    19668\n",
       "Annual Income                   19668\n",
       "Years in current job             4736\n",
       "Home Ownership                    514\n",
       "Purpose                           514\n",
       "Monthly Debt                      514\n",
       "Years of Credit History           514\n",
       "Months since last delinquent    53655\n",
       "Number of Open Accounts           514\n",
       "Number of Credit Problems         514\n",
       "Current Credit Balance            514\n",
       "Maximum Open Credit               516\n",
       "Bankruptcies                      718\n",
       "Tax Liens                         524\n",
       "dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#lists the sum of null values in every column of the dataset\n",
    "data.isnull().sum()   "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Categorical Columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Loan ID', 'Customer ID', 'Loan Status', 'Term', 'Years in current job',\n",
       "       'Home Ownership', 'Purpose'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#lists the columns with categorical data\n",
    "object_train_df=data.select_dtypes(include=['object'])    \n",
    "object_train_df.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Numerical Columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Current Loan Amount', 'Credit Score', 'Annual Income', 'Monthly Debt',\n",
       "       'Years of Credit History', 'Months since last delinquent',\n",
       "       'Number of Open Accounts', 'Number of Credit Problems',\n",
       "       'Current Credit Balance', 'Maximum Open Credit', 'Bankruptcies',\n",
       "       'Tax Liens'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#lists the columns with numerical data\n",
    "num_train_df=data.select_dtypes(include=['int','float'])     \n",
    "num_train_df.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Dropping Loan Status Null Values and Labeling it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "data.dropna(subset=['Loan Status'], inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "le = preprocessing.LabelEncoder()\n",
    "data['Loan Status'] = le.fit_transform(data['Loan Status'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Target Column Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x1ff30e9bb48>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#loan status is the target column, assigned to be zero here,it gives the count of charged off people\n",
    "coffvalue = data[data['Loan Status'] == 0]['Loan Status'].count()\n",
    "#loan status is the target column, assigned to be one here,it gives the count of fully paid people\n",
    "fpaidvalue = data[data['Loan Status'] == 1]['Loan Status'].count()\n",
    "data1 = {\"Counts\":[coffvalue, fpaidvalue] }\n",
    "statusDF = pd.DataFrame(data1, index=[\"Charged Off\", \"Fully Paid\"])\n",
    "# statusDF.head()\n",
    "statusDF.plot(kind='bar', title=\"Status of the Loan\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Term column Labeling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan ID</th>\n",
       "      <th>Customer ID</th>\n",
       "      <th>Loan Status</th>\n",
       "      <th>Current Loan Amount</th>\n",
       "      <th>Term</th>\n",
       "      <th>Credit Score</th>\n",
       "      <th>Annual Income</th>\n",
       "      <th>Years in current job</th>\n",
       "      <th>Home Ownership</th>\n",
       "      <th>Purpose</th>\n",
       "      <th>Monthly Debt</th>\n",
       "      <th>Years of Credit History</th>\n",
       "      <th>Months since last delinquent</th>\n",
       "      <th>Number of Open Accounts</th>\n",
       "      <th>Number of Credit Problems</th>\n",
       "      <th>Current Credit Balance</th>\n",
       "      <th>Maximum Open Credit</th>\n",
       "      <th>Bankruptcies</th>\n",
       "      <th>Tax Liens</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>14dd8831-6af5-400b-83ec-68e61888a048</td>\n",
       "      <td>981165ec-3274-42f5-a3b4-d104041a9ca9</td>\n",
       "      <td>1</td>\n",
       "      <td>445412.0</td>\n",
       "      <td>0</td>\n",
       "      <td>709.0</td>\n",
       "      <td>1167493.0</td>\n",
       "      <td>8 years</td>\n",
       "      <td>Home Mortgage</td>\n",
       "      <td>Home Improvements</td>\n",
       "      <td>5214.74</td>\n",
       "      <td>17.2</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>228190.0</td>\n",
       "      <td>416746.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4771cc26-131a-45db-b5aa-537ea4ba5342</td>\n",
       "      <td>2de017a3-2e01-49cb-a581-08169e83be29</td>\n",
       "      <td>1</td>\n",
       "      <td>262328.0</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>10+ years</td>\n",
       "      <td>Home Mortgage</td>\n",
       "      <td>Debt Consolidation</td>\n",
       "      <td>33295.98</td>\n",
       "      <td>21.1</td>\n",
       "      <td>8.0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>229976.0</td>\n",
       "      <td>850784.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4eed4e6a-aa2f-4c91-8651-ce984ee8fb26</td>\n",
       "      <td>5efb2b2b-bf11-4dfd-a572-3761a2694725</td>\n",
       "      <td>1</td>\n",
       "      <td>99999999.0</td>\n",
       "      <td>0</td>\n",
       "      <td>741.0</td>\n",
       "      <td>2231892.0</td>\n",
       "      <td>8 years</td>\n",
       "      <td>Own Home</td>\n",
       "      <td>Debt Consolidation</td>\n",
       "      <td>29200.53</td>\n",
       "      <td>14.9</td>\n",
       "      <td>29.0</td>\n",
       "      <td>18.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>297996.0</td>\n",
       "      <td>750090.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>77598f7b-32e7-4e3b-a6e5-06ba0d98fe8a</td>\n",
       "      <td>e777faab-98ae-45af-9a86-7ce5b33b1011</td>\n",
       "      <td>1</td>\n",
       "      <td>347666.0</td>\n",
       "      <td>1</td>\n",
       "      <td>721.0</td>\n",
       "      <td>806949.0</td>\n",
       "      <td>3 years</td>\n",
       "      <td>Own Home</td>\n",
       "      <td>Debt Consolidation</td>\n",
       "      <td>8741.90</td>\n",
       "      <td>12.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>256329.0</td>\n",
       "      <td>386958.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>d4062e70-befa-4995-8643-a0de73938182</td>\n",
       "      <td>81536ad9-5ccf-4eb8-befb-47a4d608658e</td>\n",
       "      <td>1</td>\n",
       "      <td>176220.0</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5 years</td>\n",
       "      <td>Rent</td>\n",
       "      <td>Debt Consolidation</td>\n",
       "      <td>20639.70</td>\n",
       "      <td>6.1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>15.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>253460.0</td>\n",
       "      <td>427174.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                Loan ID                           Customer ID  \\\n",
       "0  14dd8831-6af5-400b-83ec-68e61888a048  981165ec-3274-42f5-a3b4-d104041a9ca9   \n",
       "1  4771cc26-131a-45db-b5aa-537ea4ba5342  2de017a3-2e01-49cb-a581-08169e83be29   \n",
       "2  4eed4e6a-aa2f-4c91-8651-ce984ee8fb26  5efb2b2b-bf11-4dfd-a572-3761a2694725   \n",
       "3  77598f7b-32e7-4e3b-a6e5-06ba0d98fe8a  e777faab-98ae-45af-9a86-7ce5b33b1011   \n",
       "4  d4062e70-befa-4995-8643-a0de73938182  81536ad9-5ccf-4eb8-befb-47a4d608658e   \n",
       "\n",
       "   Loan Status  Current Loan Amount  Term  Credit Score  Annual Income  \\\n",
       "0            1             445412.0     0         709.0      1167493.0   \n",
       "1            1             262328.0     0           NaN            NaN   \n",
       "2            1           99999999.0     0         741.0      2231892.0   \n",
       "3            1             347666.0     1         721.0       806949.0   \n",
       "4            1             176220.0     0           NaN            NaN   \n",
       "\n",
       "  Years in current job Home Ownership             Purpose  Monthly Debt  \\\n",
       "0              8 years  Home Mortgage   Home Improvements       5214.74   \n",
       "1            10+ years  Home Mortgage  Debt Consolidation      33295.98   \n",
       "2              8 years       Own Home  Debt Consolidation      29200.53   \n",
       "3              3 years       Own Home  Debt Consolidation       8741.90   \n",
       "4              5 years           Rent  Debt Consolidation      20639.70   \n",
       "\n",
       "   Years of Credit History  Months since last delinquent  \\\n",
       "0                     17.2                           NaN   \n",
       "1                     21.1                           8.0   \n",
       "2                     14.9                          29.0   \n",
       "3                     12.0                           NaN   \n",
       "4                      6.1                           NaN   \n",
       "\n",
       "   Number of Open Accounts  Number of Credit Problems  Current Credit Balance  \\\n",
       "0                      6.0                        1.0                228190.0   \n",
       "1                     35.0                        0.0                229976.0   \n",
       "2                     18.0                        1.0                297996.0   \n",
       "3                      9.0                        0.0                256329.0   \n",
       "4                     15.0                        0.0                253460.0   \n",
       "\n",
       "   Maximum Open Credit  Bankruptcies  Tax Liens  \n",
       "0             416746.0           1.0        0.0  \n",
       "1             850784.0           0.0        0.0  \n",
       "2             750090.0           0.0        0.0  \n",
       "3             386958.0           0.0        0.0  \n",
       "4             427174.0           0.0        0.0  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# replacing the values in the column[Term] with 0 and 1 in place of short term and long term\n",
    "data['Term'].replace((\"Short Term\",\"Long Term\"),(0,1), inplace=True)\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Counts</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Short Term</th>\n",
       "      <td>72208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Long Term</th>\n",
       "      <td>27792</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Counts\n",
       "Short Term   72208\n",
       "Long Term    27792"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scount = data[data['Term'] == 0]['Term'].count()\n",
    "lcount = data[data['Term'] ==1]['Term'].count()\n",
    "\n",
    "data1 = {\"Counts\":[scount, lcount]}\n",
    "#gives the count of short and long term\n",
    "termDF = pd.DataFrame(data1, index=[\"Short Term\", \"Long Term\"])\n",
    "termDF.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "There are  19154 null values for Credit score.\n"
     ]
    }
   ],
   "source": [
    "#displays the sum of null values in credit sccore column\n",
    "print(\"There are \", data['Credit Score'].isna().sum(), \"null values for Credit score.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Scaling Credit Score Column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Applying lamda function\n",
    "data['Credit Score'] = data['Credit Score'].apply(lambda val: (val /10) if val>850 else val)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling Null values of Credit Score Column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "do_nothing = lambda: None\n",
    "cscoredf = data[data['Term']==0]\n",
    "stermAVG = cscoredf['Credit Score'].mean()\n",
    "lscoredf = data[data['Term']==1]\n",
    "ltermAVG = lscoredf['Credit Score'].mean()\n",
    "data.loc[(data.Term ==0) & (data['Credit Score'].isnull()),'Credit Score'] = stermAVG\n",
    "data.loc[(data.Term ==1) & (data['Credit Score'].isnull()),'Credit Score'] = ltermAVG"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "#For the credit score column applying conditions for the possible outcomes\n",
    "data['Credit Score'] = data['Credit Score'].apply(lambda val: \"Poor\" if np.isreal(val)\n",
    "                                                  and val < 580 else val)\n",
    "data['Credit Score'] = data['Credit Score'].apply(lambda val: \"Average\" if np.isreal(val)\n",
    "                                                  and (val >= 580 and val < 670) else val)\n",
    "data['Credit Score'] = data['Credit Score'].apply(lambda val: \"Good\" if np.isreal(val) \n",
    "                                                  and (val >= 670 and val < 740) else val)\n",
    "data['Credit Score'] = data['Credit Score'].apply(lambda val: \"Very Good\" if np.isreal(val) \n",
    "                                                  and (val >= 740 and val < 800) else val)\n",
    "data['Credit Score'] = data['Credit Score'].apply(lambda val: \"Exceptional\" if np.isreal(val) \n",
    "                                                  and (val >= 800 and val <= 850) else val)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x1ff333086c8>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# The graph lists out the counts in an ascending way\n",
    "data['Credit Score'].value_counts().sort_values(ascending = True).plot(kind='bar', title ='Number of loans in terms of Credit Score category')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Annual Income Column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "There are 19154 Missing Annual Income Values.\n"
     ]
    }
   ],
   "source": [
    "#prints the sum of null values of the column Annual Income\n",
    "print(\"There are\",data['Annual Income'].isna().sum(), \"Missing Annual Income Values.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "# By appplying mean we fill the null values\n",
    "data['Annual Income'].fillna(data['Annual Income'].mean(), inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(100000, 19)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({'Good': 75506, 'Very Good': 18479, 'Average': 6015})\n"
     ]
    }
   ],
   "source": [
    "from collections import Counter as c\n",
    "print(c(data['Credit Score']))  #returns the class count values "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Counter({1: 75506, 2: 18479, 0: 6015})"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Credit Score'] = le.fit_transform(data['Credit Score'])  #applying label encoder\n",
    "c(data['Credit Score'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Home Ownership Column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x1ff364fe8c8>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data['Home Ownership'].value_counts().sort_values(ascending = True).plot(kind='bar', title=\"Number of Loan based on Home ownership\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({'Home Mortgage': 48410, 'Rent': 42194, 'Own Home': 9182, 'HaveMortgage': 214})\n",
      "Counter({1: 48410, 3: 42194, 2: 9182, 0: 214})\n"
     ]
    }
   ],
   "source": [
    "print(c(data['Home Ownership']))\n",
    "data['Home Ownership'] = le.fit_transform(data['Home Ownership'])\n",
    "print(c(data['Home Ownership']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Years in current job"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Years in current job']=data['Years in current job'].str.extract(r\"(\\d+)\")\n",
    "data['Years in current job'] = data['Years in current job'].astype(float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "expmean = data['Years in current job'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Years in current job'].fillna(expmean, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Years in current job'].fillna(expmean, inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Dropping unwanted columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = data.drop(['Loan ID','Customer ID','Purpose'], axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Credit Problems"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Credit Problems'] = data['Number of Credit Problems'].apply(lambda x: \"No Credit Problem\" if x==0 \n",
    "                        else (\"Some Credit promblem\" if x>0 and x<5 else \"Major Credit Problems\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({'No Credit Problem': 86035, 'Some Credit promblem': 13879, 'Major Credit Problems': 86})\n",
      "Counter({1: 86035, 2: 13879, 0: 86})\n"
     ]
    }
   ],
   "source": [
    "print(c(data['Credit Problems']))\n",
    "data['Credit Problems'] = le.fit_transform(data['Credit Problems'])\n",
    "print(c(data['Credit Problems']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Credit Age"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Credit Age'] = data['Years of Credit History'].apply(lambda x: \"Short Credit Age\" if x<5 \n",
    "                                else (\"Good Credit Age\" if x>5 and x<17 else \"Exceptional Credit Age\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({'Exceptional Credit Age': 49958, 'Good Credit Age': 49848, 'Short Credit Age': 194})\n",
      "Counter({0: 49958, 1: 49848, 2: 194})\n"
     ]
    }
   ],
   "source": [
    "print(c(data['Credit Age']))\n",
    "data['Credit Age'] = le.fit_transform(data['Credit Age'])\n",
    "print(c(data['Credit Age']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = data.drop(['Months since last delinquent','Number of Open Accounts',\n",
    "                  'Maximum Open Credit','Current Credit Balance','Monthly Debt'],axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Tax Liens"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Tax Liens'] = data['Tax Liens'].apply(lambda x: \"No Tax Lien\" if x==0\n",
    "                                else (\"Some Tax Liens\" if x>0 and x<3 else \"Many Tax Liens\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({'No Tax Lien': 98062, 'Some Tax Liens': 1717, 'Many Tax Liens': 221})\n",
      "Counter({1: 98062, 2: 1717, 0: 221})\n"
     ]
    }
   ],
   "source": [
    "print(c(data['Tax Liens']))\n",
    "data['Tax Liens'] = le.fit_transform(data['Tax Liens'])\n",
    "print(c(data['Tax Liens']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bankruptcies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Bankruptcies'] = data['Bankruptcies'].apply(lambda x: \"No bankruptcies\" if x==0 \n",
    "                            else (\"Some Bankruptcies\" if x>0 and x<3 else \"Many Bankruptcies\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({'No bankruptcies': 88774, 'Some Bankruptcies': 10892, 'Many Bankruptcies': 334})\n",
      "Counter({1: 88774, 2: 10892, 0: 334})\n"
     ]
    }
   ],
   "source": [
    "print(c(data['Bankruptcies']))\n",
    "data['Bankruptcies'] = le.fit_transform(data['Bankruptcies'])\n",
    "print(c(data['Bankruptcies']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Annual Income"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "meanxoutlier = data[data['Annual Income'] < 99999999.00 ]['Annual Income'].mean()\n",
    "stddevxoutlier = data[data['Annual Income'] < 99999999.00 ]['Annual Income'].std()\n",
    "poorline = meanxoutlier -  stddevxoutlier\n",
    "richline = meanxoutlier + stddevxoutlier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Annual Income'] = data['Annual Income'].apply(lambda x: \"Low Income\" if x<=poorline \n",
    "                            else (\"Average Income\" if x>poorline and x<richline else \"High Income\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({'Average Income': 86004, 'High Income': 9145, 'Low Income': 4851})\n",
      "Counter({0: 86004, 1: 9145, 2: 4851})\n"
     ]
    }
   ],
   "source": [
    "print(c(data['Annual Income']))\n",
    "data['Annual Income'] = le.fit_transform(data['Annual Income'])\n",
    "print(c(data['Annual Income']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Current Loan Amount"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "126051.43019084723 498575.76557037106\n"
     ]
    }
   ],
   "source": [
    "lmeanxoutlier = data[data['Current Loan Amount'] < 99999999.00 ]['Current Loan Amount'].mean()\n",
    "lstddevxoutlier = data[data['Current Loan Amount'] < 99999999.00 ]['Current Loan Amount'].std()\n",
    "lowrange = lmeanxoutlier - lstddevxoutlier\n",
    "highrange = lmeanxoutlier + lstddevxoutlier\n",
    "print(lowrange, highrange)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Current Loan Amount'] = data['Current Loan Amount'].apply(lambda x: \"Small Loan\" if x<=lowrange \n",
    "                            else (\"Medium Loan\" if x>lowrange and x<highrange else \"Big Loan\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({'Medium Loan': 60112, 'Big Loan': 26506, 'Small Loan': 13382})\n",
      "Counter({1: 60112, 0: 26506, 2: 13382})\n"
     ]
    }
   ],
   "source": [
    "print(c(data['Current Loan Amount']))\n",
    "data['Current Loan Amount'] = le.fit_transform(data['Current Loan Amount'])\n",
    "print(c(data['Current Loan Amount']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(100000, 13)"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Seperating Dependent and Independent Columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = data['Loan Status']\n",
    "X = data.drop(['Loan Status'],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan Status</th>\n",
       "      <th>Current Loan Amount</th>\n",
       "      <th>Term</th>\n",
       "      <th>Credit Score</th>\n",
       "      <th>Annual Income</th>\n",
       "      <th>Years in current job</th>\n",
       "      <th>Home Ownership</th>\n",
       "      <th>Years of Credit History</th>\n",
       "      <th>Number of Credit Problems</th>\n",
       "      <th>Bankruptcies</th>\n",
       "      <th>Tax Liens</th>\n",
       "      <th>Credit Problems</th>\n",
       "      <th>Credit Age</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>1</td>\n",
       "      <td>17.2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>1</td>\n",
       "      <td>21.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>8.0</td>\n",
       "      <td>2</td>\n",
       "      <td>14.9</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>2</td>\n",
       "      <td>12.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>3</td>\n",
       "      <td>6.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Loan Status  Current Loan Amount  Term  Credit Score  Annual Income  \\\n",
       "0            1                    1     0             1              0   \n",
       "1            1                    1     0             1              0   \n",
       "2            1                    0     0             2              1   \n",
       "3            1                    1     1             1              0   \n",
       "4            1                    1     0             1              0   \n",
       "\n",
       "   Years in current job  Home Ownership  Years of Credit History  \\\n",
       "0                   8.0               1                     17.2   \n",
       "1                  10.0               1                     21.1   \n",
       "2                   8.0               2                     14.9   \n",
       "3                   3.0               2                     12.0   \n",
       "4                   5.0               3                      6.1   \n",
       "\n",
       "   Number of Credit Problems  Bankruptcies  Tax Liens  Credit Problems  \\\n",
       "0                        1.0             2          1                2   \n",
       "1                        0.0             1          1                1   \n",
       "2                        1.0             1          1                2   \n",
       "3                        0.0             1          1                1   \n",
       "4                        0.0             1          1                1   \n",
       "\n",
       "   Credit Age  \n",
       "0           0  \n",
       "1           0  \n",
       "2           1  \n",
       "3           1  \n",
       "4           1  "
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Performing Train and test split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',\n",
       "                       max_depth=None, max_features=None, max_leaf_nodes=None,\n",
       "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "                       min_samples_leaf=1, min_samples_split=2,\n",
       "                       min_weight_fraction_leaf=0.0, presort='deprecated',\n",
       "                       random_state=None, splitter='best')"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#By using DecisionTree we are fitting the model\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "dt = DecisionTreeClassifier()\n",
    "dt.fit(X_train, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Counter({0: 6711, 1: 26289})"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred_dt =dt.predict(X_test)  #prediction\n",
    "c(y_pred_dt)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##   Creating a pickle file dumping the model in it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle    #importing the pickle file\n",
    "\n",
    "pickle.dump(dt,open('loan.pkl','wb'))    #Dumping the model into the pickle file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "! jt -tmonokai"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
