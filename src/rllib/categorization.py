import cache_guessing_game_env_impl as env
import sys
import pandas as pd
from pandas.core.arrays import numeric

#def number_of_set(x):
#  return x%2 #number_of_set = 2

  # suppose "dummy" is a dual list of 7 for example. Will be relaced later
dummy = [[1, 0, 0, 0, 0], [3, 0, 0, 0, 0], [4, 0, 0, 0, 0], [1, 0, 0, 0, 0], [5, 0, 0, 0, 0], [0, 0, 1, 0, 0], [3, 0, 0, 0, 0]]

def read_file(): # will read the files in json. I have just left this function to read 'temp.txt'

  f = open('/home/geunbae/CacheSimulator/src/temp.txt', mode='r', encoding='UTF-8')
  d=f.read()
  d
  return d

def parser_action(): # split the input into [(attacker's)addr, is_guess, is_victim, is_flush, victim_addr]

  input = pd.DataFrame(dummy)  
  input = input.astype('int')
  input.columns =['addr', 'is_guess', 'is_victim', 'is_flush', 'victim_addr']
  #input['set'] = 2
  #input['set'] = input['addr']%2
  input.assign(set = lambda x: (x['addr'])%2)
  #input['set'] = input['addr'].apply(number_of_set(x))


def get_set(): # return addr%number_of_set

  #input2 = pd.DataFrame(input)
  #input2 = input2.astype('int')
  #input2 = input.assign(set = 2)
  #input[:,'set'] = 2
  #input2[:,'set'] = 2
  #input2['set'] = input2['addr'].apply(lambda x: x% 2) 
  #input2['set'] = input2[0].apply(lambda x: x% 2) 
  #input['set'] = input['addr'].apply(lambda x: x% 2) 
  #input['set'] = input.columns=['addr']% 2
  #input2['set'] = input2[0]% 2
  #input2['set'] = input2[0].apply(lambda x: x% 2) 
  #input2['set'] = input2[0].apply(lambda x: x% 2) 
  #input['set'] = input.apply(number_of_set) 
  pass
get_set()

def get_order(): # return (addr)/number_of_set

  input_set0 = input[input.set==0]
  input_set0['order'] = input_set0['addr'].rank(method='dense',ascending=False).astype(int)
  print(input_set0)

  input_set1 = input[input.set==1]
  input_set1['order'] = input_set1['addr'].rank(method='dense',ascending=True).astype(int)
  print(input_set1)

  frames = [input_set0, input_set1]
  result = pd.concat(frames)
  output = pd.DataFrame(result)
  output =output.sort_index(axis=0, ascending=True)

get_order()

def rename_addr(): # rename the addres in the pattern based on the set and the order appeared in the pattern 
# output = [#set, #the order the address appear in the attack, is_guess, is_victim, is_flush, victim_addr]
  output = output[['set','order','is_guess', 'is_victim', 'is_flush', 'victim_addr']] 
  return output


def remove(): # remove repeated access

  return output.drop_duplicates()
  print(output)

# Defining main function
def main():
  number_of_set(x)
  read_file()
  parser_action()
  get_set()
  get_order()
  rename_addr()
  remove()

# Using the special variable 
# __name__
if __name__=="__main__":
    main()