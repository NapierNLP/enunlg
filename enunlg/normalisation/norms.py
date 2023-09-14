from collections import namedtuple
from typing import List

import abc

import regex

RegexRule = namedtuple(typename='RegexRule',
                       field_names=("match_expression", "replacement_expression"))


class AbstractNormaliser(abc.ABC):
    """Abstract base class for text normalisation, assuming we may need to pre- or post-process text.
    """

    @classmethod
    def preprocess(cls, text: str):
        return text

    @classmethod
    def postprocess(cls, text) -> str:
        return text

    @classmethod
    @abc.abstractmethod
    def normalise(cls, text: str) -> str:
        pass

    @classmethod
    def normalize(cls, text: str) -> str:
        return cls.normalise(text)


class SCLSTMNormaliser(AbstractNormaliser):
    """
    Class providing SCLSTM normalisation as implemented in `RNNNLG` rules for English texts.

    cf. https://github.com/shawnwun/RNNLG/blob/master/utils/nlp.py
    """
    replacements = [(" it's ", " it is "),
                    (" don't ", " do not "),
                    (" doesn't ", " does not "),
                    (" didn't ", " did not "),
                    (" you'd ", " you would "),
                    (" you're ", " you are "),
                    (" you'll ", " you will "),
                    (" i'm ", " i am "),
                    (" they're ", " they are "),
                    (" that's ", " that is "),
                    (" what's ", " what is "),
                    (" couldn't ", " could not "),
                    (" i've ", " i have "),
                    (" we've ", " we have "),
                    (" can't ", " cannot "),
                    (" i'd ", " i would "),
                    (" i'd ", " i would "),
                    (" aren't ", " are not "),
                    (" isn't ", " is not "),
                    (" wasn't ", " was not "),
                    (" weren't ", " were not "),
                    (" won't ", " will not "),
                    (" there's ", " there is "),
                    (" there're ", " there are "),
                    (" . . ", " . "),
                    (" restaurants ", " restaurant -s "),
                    (" hotels ", " hotel -s "),
                    (" laptops ", " laptop -s "),
                    (" cheaper ", " cheap -er "),
                    (" dinners ", " dinner -s "),
                    (" lunches ", " lunch -s "),
                    (" breakfasts ", " breakfast -s "),
                    (" expensively ", " expensive -ly "),
                    (" moderately ", " moderate -ly "),
                    (" cheaply ", " cheap -ly "),
                    (" prices ", " price -s "),
                    (" places ", " place -s "),
                    (" venues ", " venue -s "),
                    (" ranges ", " range -s "),
                    (" meals ", " meal -s "),
                    (" locations ", " location -s "),
                    (" areas ", " area -s "),
                    (" policies ", " policy -s "),
                    (" children ", " child -s "),
                    (" kids ", " kid -s "),
                    (" kidfriendly ", " kid friendly "),
                    (" cards ", " card -s "),
                    (" st ", " street "),
                    (" ave ", " avenue "),
                    (" upmarket ", " expensive "),
                    (" inpricey ", " cheap "),
                    (" inches ", " inch -s "),
                    (" uses ", " use -s "),
                    (" dimensions ", " dimension -s "),
                    (" driverange ", " drive range "),
                    (" includes ", " include -s "),
                    (" computers ", " computer -s "),
                    (" machines ", " machine -s "),
                    (" ecorating ", " eco rating "),
                    (" families ", " family -s "),
                    (" ratings ", " rating -s "),
                    (" constraints ", " constraint -s "),
                    (" pricerange ", " price range "),
                    (" batteryrating ", " battery rating "),
                    (" requirements ", " requirement -s "),
                    (" drives ", " drive -s "),
                    (" specifications ", " specification -s "),
                    (" weightrange ", " weight range "),
                    (" harddrive ", " hard drive "),
                    (" batterylife ", " battery life "),
                    (" businesses ", " business -s "),
                    (" hours ", " hour -s "),
                    (" accessories ", " accessory -s "),
                    (" ports ", " port -s "),
                    (" televisions ", " television -s "),
                    (" restrictions ", " restriction -s "),
                    (" extremely ", " extreme -ly "),
                    (" actually ", " actual -ly "),
                    (" typically ", " typical -ly "),
                    (" drivers ", " driver -s "),
                    (" teh ", " the "),
                    (" definitely ", " definite -ly "),
                    (" factors ", " factor -s "),
                    (" truly ", " true -ly "),
                    (" mostly ", " most -ly "),
                    (" nicely ", " nice -ly "),
                    (" surely ", " sure -ly "),
                    (" certainly ", " certain -ly "),
                    (" totally ", " total -ly "),
                    (" # ", " number "),
                    (" & ", " and ")]

    @classmethod
    def _rnnlg_insert_space(cls, token, text):
        sidx = 0
        while True:
            sidx = text.find(token, sidx)
            if sidx == -1:
                break
            if sidx + 1 < len(text) and regex.match('[0-9]', text[sidx - 1]) and \
                    regex.match('[0-9]', text[sidx + 1]):
                sidx += 1
                continue
            if text[sidx - 1] != ' ':
                text = text[:sidx] + ' ' + text[sidx:]
                sidx += 1
            if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
                text = text[:sidx + 1] + ' ' + text[sidx + 1:]
            sidx += 1
        return text

    @classmethod
    def _rnnlg_normalise(cls, text: str) -> str:
        # Every call to normalize() in RNNLG first removes utterance-final punctuation
        regex.sub(r' [\.\?\!]$', '', text)
        # logging.debug(text)
        # lower case every word
        text = text.lower()

        # replace white spaces in front and end
        text = regex.sub(r'^\s*|\s*$', '', text)

        # normalize phone number
        ms = regex.findall(r'\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m[0], sidx)
                if text[sidx - 1] == '(':
                    sidx -= 1
                eidx = text.find(m[-1], sidx) + len(m[-1])
                text = text.replace(text[sidx:eidx], ''.join(m))

        # replace st.
        text = text.replace(';', ',')
        text = regex.sub(r'$\/', '', text)
        text = text.replace('/', ' and ')

        # replace other special characters
        text = regex.sub(r'[\":\<>@]', '', text)
        # text = re.sub('[\":\<>@\(\)]','',text)
        text = text.replace(' - ', '')

        # insert white space before and after tokens:
        for token in ['?', '.', ',', '!']:
            text = cls._rnnlg_insert_space(token, text)

        # replace it's, does't, you'd ... etc
        text = regex.sub('^\'', '', text)
        text = regex.sub('\'$', '', text)
        text = regex.sub(r'\'\s', ' ', text)
        text = regex.sub(r'\s\'', ' ', text)
        for fromx, tox in cls.replacements:
            text = ' ' + text + ' '
            text = text.replace(fromx, tox)[1:-1]

        # insert white space for 's
        text = cls._rnnlg_insert_space('\'s', text)

        # remove multiple spaces
        text = regex.sub(' +', ' ', text)

        # concatenate numbers
        tokens: List[str] = text.split()
        i = 1
        while i < len(tokens):
            if regex.match(r'^\d+$',
                           tokens[i]) and regex.match(r'\d+$',
                                                      tokens[i - 1]):
                tokens[i - 1] += tokens[i]
                del tokens[i]
            else:
                i += 1
        text = ' '.join(tokens)
        # logging.debug(text)
        return text

    @classmethod
    def normalise(cls, text: str) -> str:
        return cls._rnnlg_normalise(text)
