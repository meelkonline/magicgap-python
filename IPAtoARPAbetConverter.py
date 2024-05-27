import os
from dotenv import load_dotenv
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper


class IPAtoARPAbetConverter:
    ipa_to_arpabet = [
        ('p', 'P'),
        ('b', 'B'),
        ('t', 'T'),
        ('d', 'D'),
        ('ʈ', 'TT'),
        ('ɖ', 'DD'),
        ('c', 'C'),
        ('ɟ', 'JJ'),
        ('k', 'K'),
        ('ɡ', 'G'),
        ('q', 'Q'),
        ('ɢ', 'GG'),
        ('ʔ', 'Q'),
        ('m', 'M'),
        ('ɱ', 'MM'),
        ('n', 'N'),
        ('ɳ', 'NN'),
        ('ɲ', 'NG'),
        ('ŋ', 'NG'),
        ('ɴ', 'NG'),
        ('ʙ', 'RR'),
        ('r', 'R'),
        ('ʀ', 'RR'),
        ('ɾ', 'D'),
        ('ɽ', 'D'),
        ('ɸ', 'F'),
        ('β', 'B'),
        ('f', 'F'),
        ('v', 'V'),
        ('θ', 'TH'),
        ('ð', 'DH'),
        ('s', 'S'),
        ('z', 'Z'),
        ('ʃ', 'SH'),
        ('ʒ', 'ZH'),
        ('ʂ', 'SH'),
        ('ʐ', 'ZH'),
        ('ç', 'CH'),
        ('ʝ', 'JH'),
        ('x', 'KH'),
        ('ɣ', 'GH'),
        ('χ', 'KH'),
        ('ʁ', 'R'),
        ('ħ', 'HH'),
        ('ʕ', 'H'),
        ('h', 'HH'),
        ('ɦ', 'HH'),
        ('ɬ', 'HL'),
        ('ɮ', 'L'),
        ('ʋ', 'V'),
        ('ɹ', 'R'),
        ('ɻ', 'R'),
        ('j', 'Y'),
        ('ɰ', 'W'),
        ('w', 'W'),
        ('ɥ', 'W'),
        ('ʍ', 'WH'),
        ('l', 'L'),
        ('ɭ', 'LL'),
        ('ʎ', 'LY'),
        ('ʟ', 'LL'),
        ('i', 'IY'),
        ('y', 'IY'),
        ('ɨ', 'IY'),
        ('ʉ', 'UH'),
        ('ɯ', 'UW'),
        ('u', 'UW'),
        ('ɪ', 'IH'),
        ('ʏ', 'IH'),
        ('ʊ', 'UH'),
        ('e', 'EY'),
        ('ø', 'ER'),
        ('ɘ', 'EY'),
        ('ɵ', 'ER'),
        ('ɤ', 'ER'),
        ('o', 'OW'),
        ('ə', 'AH'),
        ('ɚ', 'ER'),
        ('ɛ', 'EH'),
        ('œ', 'EH'),
        ('ɜ', 'ER'),
        ('ɞ', 'ER'),
        ('ʌ', 'AH'),
        ('ɔ', 'AO'),
        ('æ', 'AE'),
        ('a', 'AA'),
        ('ɶ', 'AE'),
        ('ɑ', 'AA'),
        ('ɒ', 'AA'),
        ('ɐ', 'AH'),
        ('aɪ', 'AY'),
        ('aʊ', 'AW'),
        ('eɪ', 'EY'),
        ('oʊ', 'OW'),
        ('ɔɪ', 'OY'),
        ('pʰ', 'P'),
        ('tʰ', 'T'),
        ('kʰ', 'K'),
        ('bʲ', 'B'),
        ('m̩', 'M'),
        ('n̩', 'N'),
        ('ŋ̍', 'NG'),
        ('ɹ̩', 'R'),
        ('l̩', 'L'),
        ('ɾ̃', 'R'),
        ('ɾ̩', 'R'),
        ('ɾ̩̃', 'R'),
        ('ɾ̩̃', 'R')
    ]

    ipa_to_arpabet_dict = dict(ipa_to_arpabet)

    def __init__(self, text):
        self.text = text
        load_dotenv()  # Load environment variables from a .env file
        self.APP_ENV = os.getenv('APP_ENV')
        if self.APP_ENV == "local":
            EspeakWrapper.set_library('C:\\Program Files\\eSpeak NG\\libespeak-ng.dll')
        self.ipa_string = self.phonemize_text()

    def phonemize_text(self):
        return phonemize(self.text, preserve_punctuation=False)

    def convert_ipa_to_arpabet(self):
        arpabet_string = []
        i = 0
        while i < len(self.ipa_string):
            for length in [3, 2, 1]:  # Check for the longest possible match first
                if i + length <= len(self.ipa_string) and self.ipa_string[i:i + length] in self.ipa_to_arpabet_dict:
                    arpabet_string.append(self.ipa_to_arpabet_dict[self.ipa_string[i:i + length]])
                    i += length
                    break
            else:  # No break means no match was found
                arpabet_string.append(self.ipa_string[i])  # Keep the original character if no match
                i += 1
        return ' '.join(arpabet_string)

    def get_arpabet(self):
        return self.convert_ipa_to_arpabet()
