class AudioSegmentsToVisemes:
    def __init__(self):
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
            'aa': ['AA', 'A', 'AH', 'AX'],
            'E': ['EH', 'E', 'Y'],
            'I': ['IH', 'IY', 'IX'],
            'O': ['OW', 'OY', 'AOA'],
            'U': ['UH', 'UW', 'UX']
        }

    def phonemes_to_visemes(self, phonemes):
        visemes = []
        for phoneme in phonemes.split():
            found = False
            for viseme, values in self.viseme_to_arpabet.items():
                if phoneme.upper() in values:
                    visemes.append(viseme)
                    found = True
                    break
            if not found:
                print(f'Phoneme not found: {phoneme}')
        return visemes

    def get_next_viseme(self, results, current_frame):
        for frame, values in sorted(results.items()):
            if frame <= current_frame:
                continue
            for viseme, presence in values.items():
                if presence == 1:
                    return viseme, frame - current_frame
        return None, None

    def process_visemes(self, segments):
        results = {}
        possible_phonemes = {viseme: 0 for viseme in self.viseme_to_arpabet}
        precision = 4
        multiplier = 1000
        blend_offset = 150

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
                    results[frame][viseme] = 1

            else:
                frame = int(round(round(segment['start'], precision) * multiplier))
                results[frame] = possible_phonemes.copy()
                results[frame]['sil'] = 1

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
                    for j in range(1, next_viseme_distance):
                        final[frame + j] = values.copy()

                        for viseme in possible_phonemes:
                            final[frame + j][viseme] = values[viseme] * (1.0 - step * j)
                        if current_viseme != 'sil':  # Ensure current viseme fades out
                            final[frame + j][current_viseme] = 1.0 - step * j
                        final[frame + j][next_viseme] = step * j

        scaled_final_list = [[int(value * 1000) for value in frame_data.values()] for frame_data in final.values()]

        return scaled_final_list

