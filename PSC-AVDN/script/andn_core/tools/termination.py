"""Recognize complete stop-only instructions; never infer completion from IDs or GT."""
import re


def stop_only_instruction(dialogue):
    text = re.sub(r'\[ins\]', '', str(dialogue), flags=re.I).strip()
    return bool(re.fullmatch(
        r"(?:you (?:are|have) already (?:at|reached) (?:the |your )?destination|"
        r"you are at (?:the |your )?destination|stop here|stay here)[.!\s]*",
        text, flags=re.I))


def stop_step(dialogue):
    if not stop_only_instruction(dialogue):
        raise ValueError('Not a complete stop-only instruction')
    return dict(instruction=str(dialogue).strip(), move_dir='Stop', dest='current position')
