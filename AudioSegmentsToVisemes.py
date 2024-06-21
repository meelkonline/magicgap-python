import re

from bibtexparser import splitter


class AudioSegmentsToVisemes:
    def __init__(self, lang):
        self.lang = lang
        self.viseme_to_arpabet = {
            'sil': ['sil', '-', 'PAU', 'EPI', 'None'],
            'PP': ['P', 'B', 'M', 'BCL', 'PCL', 'EM', 'AE', 'AW', 'W'],
            'FF': ['F', 'V'],
            'TH': ['TH', 'DH'],
            'DD': ['T', 'D', 'DX', 'DCL', 'TCL', 'L', 'EL'],
            'kk': ['K', 'G', 'GCL', 'KCL', 'NX', 'NG', 'AO'],
            'CH': ['CH', 'JH', 'SH'],
            'SS': ['S', 'Z', 'ZH'],
            'nn': ['N', 'L', 'EN', 'EL', 'NX', 'ENG'],
            'RR': ['R', 'ER', 'AXR'],
            'aa': ['AA', 'A', 'AH', 'AX', 'AY', 'H', 'HH'],
            'E': ['EH', 'E', 'Y', 'EY'],
            'I': ['IH', 'IY', 'IX'],
            'O': ['OW', 'OY', 'AOA'],
            'U': ['UH', 'UW', 'UX']
        }

        self.viseme_to_ipa = {
            'sil': ['—', '-'],
            'PP': ['p', 'b', 'm', 'b̚', 'p̚', 'm̩', 'æ', 'aʊ', 'w'],
            'FF': ['f', 'v'],
            'TH': ['θ', 'ð'],
            'DD': ['t', 'd', 'ɾ', 'd̚', 't̚', 'l', 'l̩'],
            'kk': ['k', 'ɡ', 'ɡ̚', 'k̚', 'ɾ̃', 'ŋ', 'ɔ'],
            'CH': ['tʃ', 'dʒ', 'ʃ'],
            'SS': ['s', 'z', 'ʒ'],
            'nn': ['n', 'l', 'n̩', 'l̩', 'ɾ̃', 'ŋ̍'],
            'RR': ['ɹ', 'ɝ', 'ɚ'],
            'aa': ['ɑ', 'ɒ', 'A', 'ʌ', 'ə', 'aɪ', 'h', 'ɑ̃'],
            'E': ['ɛ', 'ɛ', 'j', 'eɪ', 'ɛ̃'],
            'I': ['ɪ', 'i', 'ɨ'],
            'O': ['oʊ', 'ɔɪ', 'œ̃', 'ɔ̃'],
            'U': ['ʊ', 'u', 'ʉ']
        }

        # Create a flattened list of all phonemes
        all_phonemes = [phoneme for sublist in self.viseme_to_ipa.values() for phoneme in sublist]
        # Sort phonemes by length to prioritize longer phonemes in regex matching
        all_phonemes_sorted = sorted(set(all_phonemes), key=len, reverse=True)
        # Create a regex pattern that matches any of the phonemes
        self.phoneme_pattern = re.compile('|'.join(re.escape(phoneme) for phoneme in all_phonemes_sorted))

    def split_ipa(self, ipa_text):
        return self.phoneme_pattern.findall(ipa_text)

    def phonemes_to_visemes(self, phonemes):
        visemes = []
        phoneme_list = self.split_ipa(phonemes)
        for phoneme in phoneme_list:
            if self.find_and_add_ipa_viseme(phoneme, visemes):
                continue

            print("not found" + phoneme)

        return visemes

    def find_and_add_ipa_viseme(self, phoneme, visemes):
        found = False
        for viseme, values in self.viseme_to_ipa.items():
            if phoneme in values:
                visemes.append(viseme)
                found = True
                break
        return found

    def find_and_add_viseme(self, phoneme, visemes):
        found = False
        phoneme_upper = phoneme.upper()
        for viseme, values in self.viseme_to_ipa.items():
            if phoneme_upper in values:
                visemes.append(viseme)
                found = True
                break
        return found

    def get_next_viseme(self, results, current_frame):
        for frame, values in sorted(results.items()):
            if frame <= current_frame:
                continue
            for viseme, presence in values.items():
                if presence == 1:
                    return viseme, frame - current_frame
        return None, None

    def get_factors(self, viseme):
        # Mapping from visemes to their influencing visemes with factors
        coarticulation_rules = {
            'en': {
                'sil': [],
                'PP': [('FF', 0.25)],
                'FF': [('PP', 0.25)],
                'TH': [('SS', 0.25)],
                'DD': [('nn', 0.33), ('SS', 0.25)],
                'kk': [('nn', 0.33)],
                'CH': [('SS', 0.33)],
                'SS': [('DD', 0.33), ('CH', 0.33)],
                'nn': [('kk', 0.33), ('DD', 0.25)],
                'RR': [('aa', 0.25)],
                'aa': [('RR', 0.25)],
                'E': [('I', 0.20), ('aa', 0.20)],
                'I': [('E', 0.20)],
                'O': [('U', 0.20)],
                'U': [('O', 0.20)]
            },
            'fr': {
                'sil': [],
                'PP': [('FF', 0.30)],  # Example adaptations for French
                'FF': [('PP', 0.30)],
                'TH': [('SS', 0.20)],
                'DD': [('nn', 0.35), ('SS', 0.20)],
                'kk': [('nn', 0.35)],
                'CH': [('SS', 0.30)],
                'SS': [('DD', 0.35), ('CH', 0.30)],
                'nn': [('kk', 0.35), ('DD', 0.30)],
                'RR': [('aa', 0.20)],
                'aa': [('RR', 0.20)],
                'E': [('I', 0.25), ('aa', 0.25)],
                'I': [('E', 0.25)],
                'O': [('U', 0.25)],
                'U': [('O', 0.25)]
            }
        }

        coarticulation_rule = coarticulation_rules[self.lang]

        return coarticulation_rule.get(viseme, [])

    def adjust_next_viseme_intensity(self, current_viseme, next_viseme):
        # Define relationships or proximity between visemes
        proximity_maps = {
            'en': {
                'PP': {'FF': 0.8, 'TH': 0.5, 'DD': 0.7},  # Example: 'PP' transitions smoothly to 'FF'
                'FF': {'PP': 0.8, 'TH': 0.6},
                'TH': {'FF': 0.6, 'DD': 0.5},
                'DD': {'TH': 0.5, 'SS': 0.4, 'nn': 0.7},  # 'DD' and 'nn' share similar tongue placements
                'kk': {'nn': 0.5, 'CH': 0.4},  # Velar sounds can lead into ch/sh sounds or nasal sounds
                'CH': {'SS': 0.8, 'kk': 0.4},  # Post-alveolar affricates/fricatives are close to sibilants
                'SS': {'DD': 0.4, 'CH': 0.8},  # Sibilants share articulation locations
                'nn': {'DD': 0.7, 'kk': 0.5},  # Nasal and alveolar/velar sounds
                'RR': {'aa': 0.5},  # Rhotics can influence open vowels
                'aa': {'RR': 0.5, 'E': 0.3},  # Open vowels transitioning to rhotics or front vowels
                'E': {'I': 0.6, 'aa': 0.3},  # Front vowels are close in articulation
                'I': {'E': 0.6, 'O': 0.4},  # High front to mid-back vowel transition
                'O': {'U': 0.7, 'I': 0.4},  # Rounded vowels transition smoothly
                'U': {'O': 0.7}  # Close back vowels to mid-back vowels
            },
            'fr': {
                'PP': {'FF': 0.7, 'TH': 0.6, 'DD': 0.8},  # Example adaptations for French
                'FF': {'PP': 0.7, 'TH': 0.5},
                'TH': {'FF': 0.5, 'DD': 0.4},
                'DD': {'TH': 0.4, 'SS': 0.3, 'nn': 0.8},  # 'DD' and 'nn' share similar tongue placements
                'kk': {'nn': 0.6, 'CH': 0.5},  # Velar sounds can lead into ch/sh sounds or nasal sounds
                'CH': {'SS': 0.7, 'kk': 0.5},  # Post-alveolar affricates/fricatives are close to sibilants
                'SS': {'DD': 0.3, 'CH': 0.7},  # Sibilants share articulation locations
                'nn': {'DD': 0.8, 'kk': 0.6},  # Nasal and alveolar/velar sounds
                'RR': {'aa': 0.6},  # Rhotics can influence open vowels
                'aa': {'RR': 0.6, 'E': 0.4},  # Open vowels transitioning to rhotics or front vowels
                'E': {'I': 0.5, 'aa': 0.4},  # Front vowels are close in articulation
                'I': {'E': 0.5, 'O': 0.3},  # High front to mid-back vowel transition
                'O': {'U': 0.6, 'I': 0.3},  # Rounded vowels transition smoothly
                'U': {'O': 0.6}  # Close back vowels to mid-back vowels
            }
        }

        proximity_map = proximity_maps[self.lang]

        # Default to full intensity if no specific proximity is defined
        base_intensity = 1.0
        if current_viseme in proximity_map and next_viseme in proximity_map[current_viseme]:
            base_intensity = proximity_map[current_viseme][next_viseme]

        return base_intensity

    def process_visemes(self, segments):
        results = {}
        possible_phonemes = {viseme: 0.0 for viseme in self.viseme_to_ipa}
        precision = 3
        multiplier = 100
        blend_offset = 10

        for segment in segments:
            if segment['type'] == 'speech' and segment['data']:
                visemes = self.phonemes_to_visemes(segment['data'])
                num = len(visemes)
                duration = segment['end'] - segment['start']
                step = duration / num

                for i, viseme in enumerate(visemes):
                    frame = int((round(segment['start'] + (step * i), precision) * multiplier))
                    if frame not in results:
                        results[frame] = possible_phonemes.copy()
                    results[frame][viseme] = 1.0

            else:
                frame = int(round(round(segment['start'], precision) * multiplier))
                results[frame] = possible_phonemes.copy()
                results[frame]['sil'] = 1.0

        last_frame = max(results.keys())
        final = {}

        for frame in sorted(results.keys()):
            values = results[frame]
            current_viseme = next((viseme for viseme, presence in values.items() if presence == 1), 'sil')

            # Warm up frames: Start first lip movement
            if frame == 0:
                for i in range(blend_offset, 0, -1):
                    final[frame - i] = possible_phonemes.copy()
                    final[frame - i][current_viseme] = max(0.0, 1.0 - i / blend_offset)

            # Current Frame
            final[frame] = values.copy()

            if frame == last_frame:
                # Last frame: end the last lip movement
                for i in range(1, blend_offset + 1):
                    final[frame + i] = possible_phonemes.copy()
                    final[frame + i][current_viseme] = max(0.0, 1.0 - (i / blend_offset))

            else:
                next_viseme, next_viseme_distance = self.get_next_viseme(results, frame)
                if next_viseme_distance > 0:
                    step = 1 / next_viseme_distance
                    target_intensity = self.adjust_next_viseme_intensity(current_viseme, next_viseme)
                    for j in range(0, next_viseme_distance):
                        final[frame + j] = values.copy()
                        for viseme in possible_phonemes:
                            final[frame + j][viseme] = values[viseme] * (1.0 - step * j)
                        if current_viseme != 'sil':  # Ensure current viseme fades out, if different from previous
                            if current_viseme != next_viseme:
                                final[frame + j][current_viseme] = 1.0 - step * j
                                final[frame + j][next_viseme] = step * j * target_intensity
                                decrease_factors = self.get_factors(current_viseme)
                                increase_factors = self.get_factors(next_viseme)
                                for co_viseme, co_factor in decrease_factors:
                                    final[frame + j][co_viseme] = max(final[frame + j][co_viseme],
                                                                      (1.0 - step * j) * co_factor)
                                for co_viseme, co_factor in increase_factors:
                                    value = max(final[frame + j][co_viseme],
                                                                      round((step * j) * co_factor * target_intensity,
                                                                            3))
                                    final[frame + j][co_viseme] = value

                            else:
                                # No transitions: the same viseme is used continuously
                                final[frame + j][current_viseme] = 1.0
                                increase_factors = self.get_factors(current_viseme)
                                for co_viseme, co_factor in increase_factors:
                                    final[frame + j][co_viseme] = max(final[frame + j][co_viseme], co_factor)

        scaled_final_list = [[int(value * 1000) for value in frame_data.values()] for frame_data in final.values()]

        return scaled_final_list
