# Useful helper functions

converter = {1:'A',2:'G',3:'C',4:'T',
             'A':1,'G':2,'C':3,'T':4}


def num_to_str(num_seq):
    """Converts a list of numbers to its corresponding bases
    i.e) 1,2,3,4 => 'A','G','T','C'
    """
    str_seq = []
    for i in num_seq:
        str_seq.append(converter.get(i))
    return str_seq


def str_to_num(str_seq):
    """Converts a list of numbers to its corresponding bases
    i.e) 'A','G','T','C' => 1,2,3,4
    """
    num_seq = []
    for i in str_seq:
        num_seq.append(converter.get(i))
    return num_seq
