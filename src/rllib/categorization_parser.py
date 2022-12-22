import sys
import pandas as pd
from cache_guessing_game_env_wrapper import CacheGuessingGameEnvWrapper as CacheGuessingGameEnv
from pandas.core.arrays import numeric

class CategorizationParser:
  def __init__(self, number_of_set=2, attacker_address_range_max=8):
    self.attacker_address_range_max = attacker_address_range_max #not include, the max value of addresses
    self.number_of_set = number_of_set

  def __init__(self, env=None):
    self.gameenv = env

  def _get_set(self, row): 
    """return set number"""
    return row['addr']%self.number_of_set

  def readfile(self,filename):# python categorization_parser.py temp.txt
    patterns=[]
    f = open(filename, mode='r', encoding='UTF-8')
    lines = f.readlines()
    for l in lines:
      l = l.split()
      l = [int(i) for i in l]
      patterns.append(l)
    return patterns

  def parse_action(self, action): 
    #gameenv = CacheGuessingGameEnv() 
    action = self.gameenv.parse_action(action)
    return action

  def convert_dataframe(self, input): # split into [(attacker's)addr, is_guess, is_victim, is_flush, victim_addr]
    df = pd.DataFrame(input)  
    df = df.astype('int')
    df.columns =['addr', 'is_guess', 'is_victim', 'is_flush', 'victim_addr']
    return df


  def add_set_column(self, df):
    df['set'] = df.apply (lambda row: self._get_set(row), axis=1)
    return df
    
  def _make_order(self, df, col_name):
    """return the order of each element in df[col_name]"""
    order = [-1 for i in range(self.attacker_address_range_max)] # could be further optimzed if min address is not 0
    cnt = 0
    for index, row in df.iterrows():
      value = row[col_name]
      if order[value] == -1:
        order[value] = cnt
        cnt = cnt + 1
    #print(f'order = {order}')
    return order

  def _get_order(self, row, col_name, order): 
    """return the order of each element in df[col_name]"""
    return order[row[col_name]]

  def rename_column(self,df, col_name):
    """rename the column based on the order the item appear in the column"""
    order = self._make_order(df, col_name)
    new_col_name = col_name + '_renamed'
    df[new_col_name] = df.apply (lambda row: self._get_order(row, col_name, order), axis=1)
    return df

  def is_same_action_df(self, action1, action2):
    if action1['is_victim'] == action2['is_victim'] and action1['is_victim'] == 1: # If both are is_victim==true, ignore rest of the columns
      return True
    if action1['is_victim'] != action2['is_victim']:
      return False
    if action1['is_guess'] == action2['is_guess'] and action1['is_guess'] == 1: # If both are is_guess==true, ignore rest of the columns
      return True
    if action1['is_guess'] != action2['is_guess']:
      return False
    if action1['addr_renamed'] == action2['addr_renamed'] and action1['set_renamed']== action2['set_renamed']: # else match the address and set
      return True
    return False

  def remove_rep(self, df): 
    """remove contiguous repeated access"""
    for index_i, row in df.iterrows():
      if index_i != 0:
        if self.is_same_action_df(last_row, row):
          df = df.drop(index=index_i, axis=0)
      last_row=row
    return df

  def is_same_action_list(self, action1, action2):
    """ action format [is_guess,  is_victim,  is_flush,  victim_addr,  addr_renamed,  set_renamed]"""
    if action1[1] == action2[1] and action1[1] == 1: # If both are is_victim==true, ignore rest of the columns
      return True
    if action1[1] != action2[1]: #  is_victim is different
      return False
    if action1[0] == action2[0] and action1[0] == 1: # If both are is_guess==true, ignore rest of the columns
      return True
    if action1[0] == action2[0]: # is_guess is different 
      return False
    if action1[4] == action2[4] and action1[5]== action2[5]: # else match the address and set
      return True
    return False

  def is_same_base_pattern(self,pattern1, pattern2):
    """ return whether two patterns after renaming is the same"""
    if len(pattern1) != len(pattern2):
      return False
    for i in range(len(pattern1)):
      if self.is_same_action_list(pattern1[i],pattern2[i]) == False:
        return False
    return True

  def main_parser(self, pattern):
    """output a pattern after renaming,
    format [is_guess,  is_victim,  is_flush,  victim_addr,  addr_renamed,  set_renamed]"""
    pattern_parsed = []
    for action in pattern :
      action_parsed = self.parse_action(action)
      pattern_parsed.append(action_parsed)
    df = self.convert_dataframe(pattern_parsed)
    print(df)
    df = self.add_set_column(df)
    df = self.rename_column(df, 'addr') # rename address
    df = self.rename_column(df, 'set') # rename set
    print(df)
    df = self.remove_rep(df) #remove repeated action
    df = df.drop(columns=['addr', 'set'], axis=1)
    print(df)
    output = df.values.tolist()
    return output

def main(argv): # Defining main function
  filename = argv[1]
  print(filename)

  categorization_parser = CategorizationParser()
  patterns = categorization_parser.readfile(filename)
  print(patterns)
  
  base_pattern = categorization_parser.main_parser(patterns[0])
  print(base_pattern)
  #for pattern in patterns :
  #  base_pattern = categorization_parser.main_parser(pattern)

if __name__=="__main__": # Using the special variable
    main(sys.argv)