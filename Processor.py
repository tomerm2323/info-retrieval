import re
from nltk.corpus import stopwords

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

class Processor:
  def __init__(self):
    pass
  def patterns(self):
    def get_date_pattern():
      m_short = "(Jan|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
      m_full =  "(January|March|April|May|June|July|August|September|October|November|December)"
      format1 = r"\b((Jan|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s([0-9]|([1-2][0-9])|3[0-1]),\s\d{4})\b" # example: jan 26, 2020
      format2 = r"\b((January|March|April|May|June|July|August|September|October|November|December)\s([0-9]|([1-2][0-9])|3[0-1]),\s\d{4})\b" # example: January 26, 2020
      format3 = r"\b(([0-9]|([1-2][0-9])|3[0-1])\s(Jan|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4})\b" # example: 9 oct 2021
      format4=  r"\b(([0-9]|([1-2][0-9])|3[0-1])\s(January|March|April|May|June|July|August|September|October|November|December)\s\d{4})\b" # example: 9 october 2021
      # format5 = r"\b(?<=\s)\b\d{4}(-|\.|/)0[1 3 4 5 6 7 8 9]|1[0-2](-|\.|/)0[1-9]|1[0-9]|2[0-9]|3[0-1]\b"# exapmle: 2021-03-31 or  2021.03.31  or  2021/03/31 (yy mm dd)
      # format6 = # example: 31-03-2021 or 31.03.2021 or 31/03/2021  (dd mm yy)
      # format7 = # exapmle: 03-31-2021 or 03.31.2021 or 03/31/2021 (mm dd yy)
      feb_format1= r"\b((Feb|February)([0-9]|1[0-9]|2[0-8])\s,\s\d{4})\b" # example: feb 26, 2020
      feb_format2 = r"\b(([0-9]|1[0-9]|2[0-8])\s(Feb|February)\s\d{4})\b" # example: 9 February 2021
      reg_exp = f"{format1}|{format2}|{format3}|{format4}|{feb_format1}|{feb_format2}"
      return reg_exp
    def get_time_pattern():
      format1 = r"\b(([0-9]|1[0-2])\.[0-5][0-9](PM|AM))\b" # hh/./mm/PM\AM
      format2 = r"\b([0-9]|1[0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])\b" ## hh/:/mm/:/ss
      format3 = r"\b(0[0-9]|1[0-2])([0-5][0-9])((a|p)\.m\.)"
      reg_exp = f"{format1}|{format2}|{format3}"
      return reg_exp
    def get_percent_pattern():
      reg_exp = r"(?<=\s)\b[0-9]+\b%(?=\s|.|\n|!)"
      return reg_exp
    def get_number_pattern():
      format1 = r"^(\+|-)?\b[0-9]+(\.\d{3}|,\d{3}|[0-9])*([\.]\d+)?\b(?!%|percent)((?!,\w|\.\w|[A-Z]))"
      format2= r"(?<=\s|\()(\+|-)?\b[0-9]+(\.\d{3}|,\d{3}|[0-9])*([\.]\d+)?\b(?!%|percent)((?!,\w|\.\w|[A-Z]))"
      reg_exp = f"{format1}|{format2}"
      return reg_exp 
    def get_word_pattern():
      format1 = r"^\b(([a-z]|[A-Z])+-([a-z]|[A-Z])+)*([a-z]|[A-Z])+('([a-z]|[A-Z])+)?\b"
      format2 = r"(?<=\s)\b(([a-z]|[A-Z])+-([a-z]|[A-Z])+)*([a-z]|[A-Z])+('([a-z]|[A-Z])+)?\b'?"
      format3 = r"(?<=\s)\b([a-z]|[A-Z])+\b'"
      reg_exp = f"{format1}|{format2}|{format3}"
      return reg_exp   
    RE_TOKENIZE = re.compile(rf"""
    (                                
    # dates
    |(?P<DATE>{get_date_pattern()})
    # time
    |(?P<TIME>{get_time_pattern()})
    # Percents
    |(?P<PERCENT>{get_percent_pattern()})
    # Numbers
    |(?P<NUMBER>{get_number_pattern()})
    # Words
    |(?P<WORD>{get_word_pattern()})
    # space
    |(?P<SPACE>[\s\t\n]+) 
    # everything else
    |(?P<OTHER>.))""",  re.MULTILINE | re.IGNORECASE | re.VERBOSE | re.UNICODE)
    return RE_TOKENIZE

  def tokenize(self, text):
    print("Proccessor -> tokenize : Started")
    
    preTokens =  [token.group() for token in RE_WORD.finditer(text.lower())] 
      
    tokens = []
    for t in preTokens:
      if t != "?":
        if t not in all_stopwords:
          tokens.append(t)
        
    return tokens
    
