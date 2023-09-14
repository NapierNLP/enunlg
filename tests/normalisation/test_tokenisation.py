import regex


from enunlg.normalisation.tokenisation import TGenTokeniser


SAMPLE_TEXT = """
Five o'clock Ed Loyce washed up, tossed on his hat and coat, got his car
out and headed across town toward his TV sales store. He was tired. His
back and shoulders ached from digging dirt out of the basement and
wheeling it into the back yard. But for a forty-year-old man he had done
okay. Janet could get a new vase with the money he had saved; and he
liked the idea of repairing the foundations himself!

It was getting dark. The setting sun cast long rays over the scurrying
commuters, tired and grim-faced, women loaded down with bundles and
packages, students swarming home from the university, mixing with clerks
and businessmen and drab secretaries. He stopped his Packard for a red
light and then started it up again. The store had been open without him;
he'd arrive just in time to spell the help for dinner, go over the
records of the day, maybe even close a couple of sales himself. He drove
slowly past the small square of green in the center of the street, the
town park. There were no parking places in front of LOYCE TV SALES AND
SERVICE. He cursed under his breath and swung the car in a U-turn. Again
he passed the little square of green with its lonely drinking fountain
and bench and single lamppost.

From the lamppost something was hanging. A shapeless dark bundle,
swinging a little with the wind. Like a dummy of some sort. Loyce rolled
down his window and peered out. What the hell was it? A display of
some kind? Sometimes the Chamber of Commerce put up displays in the
square.

Again he made a U-turn and brought his car around. He passed the park
and concentrated on the dark bundle. It wasn't a dummy. And if it was a
display it was a strange kind. The hackles on his neck rose and he
swallowed uneasily. Sweat slid out on his face and hands.

It was a body. A human body.

       *       *       *       *       *

"Look at it!" Loyce snapped. "Come on out here!"

Don Fergusson came slowly out of the store, buttoning his pin-stripe
coat with dignity. "This is a big deal, Ed. I can't just leave the guy
standing there."

"See it?" Ed pointed into the gathering gloom. The lamppost jutted up
against the sky--the post and the bundle swinging from it. "There it is.
How the hell long has it been there?" His voice rose excitedly. "What's
wrong with everybody? They just walk on past!"

Don Fergusson lit a cigarette slowly. "Take it easy, old man. There must
be a good reason, or it wouldn't be there."

"A reason! What kind of a reason?"

Fergusson shrugged. "Like the time the Traffic Safety Council put that
wrecked Buick there. Some sort of civic thing. How would I know?"

Jack Potter from the shoe shop joined them. "What's up, boys?"

"There's a body hanging from the lamppost," Loyce said. "I'm going to
call the cops."

"They must know about it," Potter said. "Or otherwise it wouldn't be
there."

"I got to get back in." Fergusson headed back into the store. "Business
before pleasure."

Loyce began to get hysterical. "You see it? You see it hanging there? A
man's body! A dead man!"

"Sure, Ed. I saw it this afternoon when I went out for coffee."

"You mean it's been there all afternoon?"

"Sure. What's the matter?" Potter glanced at his watch. "Have to run.
See you later, Ed."

Potter hurried off, joining the flow of people moving along the
sidewalk. Men and women, passing by the park. A few glanced up curiously
at the dark bundle--and then went on. Nobody stopped. Nobody paid any
attention.

"I'm going nuts," Loyce whispered. He made his way to the curb and
crossed out into traffic, among the cars. Horns honked angrily at him.
He gained the curb and stepped up onto the little square of green.
"""


def original_tgen_tokenization(text):
    """
    Tokenize the given text (i.e., insert spaces around all tokens)

    Copied from `tgen/futil.py`, which is under the Apache 2 License.

    See https://github.com/UFAL-DSG/tgen for more details.
    """
    toks = ' ' + text + ' '  # for easier regexes

    # enforce space around all punct
    toks = regex.sub(r'(([^\p{IsAlnum}\s\.\,−\-])\2*)', r' \1 ', toks)  # all punct (except ,-.)
    toks = regex.sub(r'([^\p{N}])([,.])([^\p{N}])', r'\1 \2 \3', toks)  # ,. & no numbers
    toks = regex.sub(r'([^\p{N}])([,.])([\p{N}])', r'\1 \2 \3', toks)  # ,. preceding numbers
    toks = regex.sub(r'([\p{N}])([,.])([^\p{N}])', r'\1 \2 \3', toks)  # ,. following numbers
    toks = regex.sub(r'(–-)([^\p{N}])', r'\1 \2', toks)  # -/– & no number following
    toks = regex.sub(r'(\p{N} *|[^ ])(-)', r'\1\2 ', toks)  # -/– & preceding number/no-space
    toks = regex.sub(r'([-−])', r' \1', toks)  # -/– : always space before

    # keep apostrophes together with words in most common contractions
    toks = regex.sub(r'([\'’´]) (s|m|d|ll|re|ve)\s', r' \1\2 ', toks)  # I 'm, I 've etc.
    toks = regex.sub(r'(n [\'’´]) (t\s)', r' \1\2 ', toks)  # do n't

    # other contractions, as implemented in Treex
    toks = regex.sub(r' ([Cc])annot\s', r' \1an not ', toks)
    toks = regex.sub(r' ([Dd]) \' ye\s', r' \1\' ye ', toks)
    toks = regex.sub(r' ([Gg])imme\s', r' \1im me ', toks)
    toks = regex.sub(r' ([Gg])onna\s', r' \1on na ', toks)
    toks = regex.sub(r' ([Gg])otta\s', r' \1ot ta ', toks)
    toks = regex.sub(r' ([Ll])emme\s', r' \1em me ', toks)
    toks = regex.sub(r' ([Mm])ore\'n\s', r' \1ore \'n ', toks)
    toks = regex.sub(r' \' ([Tt])is\s', r' \'\1 is ', toks)
    toks = regex.sub(r' \' ([Tt])was\s', r' \'\1 was ', toks)
    toks = regex.sub(r' ([Ww])anna\s', r' \1an na ', toks)

    # clean extra space
    toks = regex.sub(r'\s+', ' ', toks)
    toks = toks.strip()
    return toks


tgen_tokenization = original_tgen_tokenization(SAMPLE_TEXT)

def test_tgen_tokenization():
    assert TGenTokeniser.tokenise(SAMPLE_TEXT) == tgen_tokenization
