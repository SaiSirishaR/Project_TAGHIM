from symbols import symbols


#print ("symbols are:",  [s for s in symbols])
_symbol_to_id = {s: i for i, s in enumerate(symbols)}



def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _should_keep_symbol(s):
 return s in _symbol_to_id and s is not '_' and s is not '~'


def text_to_seq(text):
 sequence = []

 #sequence += _symbols_to_sequence(_clean_text(text))
 sequence +=_symbols_to_sequence(text)
 print("seq is", sequence)

file = open('/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/txt.done.data')
for line in file:
  print("data is", line)

  text_to_seq(line)

