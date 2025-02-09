def is_all_caps(line):
    return line.replace('\n', '').strip().isupper()


def merge_lines(lines):
    processed_lines = []
    i = 0

    while i < len(lines):
        current_line = lines[i].strip()
        
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
        else:
            next_line = ''

        if (
            is_all_caps(current_line) or
            is_all_caps(next_line) or
            current_line.endswith('.') or
            len(current_line) == 0 or  # Current line is empty
            next_line[:1].isupper() or
            next_line.replace('"', '')[:1].isupper()
        ):
            processed_lines.append(current_line + '\n')
        else:
            # if current_line:
            #     processed_lines.append(current_line + ' ')
            # else:
            processed_lines.append(current_line + ' ')

        i += 1

    return processed_lines