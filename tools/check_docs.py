"""Check the portfolio README, documentation links, and local image assets."""
import argparse
import json
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlsplit


def check(root, inventory=None):
    root = root.resolve()
    known = set(inventory or [])
    errors = []
    documents = [root / 'README.md', *sorted((root / 'docs').rglob('*.md'))]
    for document in documents:
        if not document.is_file():
            errors.append(f'Missing document: {document.relative_to(root)}')
            continue
        text = document.read_text(encoding='utf-8')
        # Inline code and fenced examples are not Markdown link destinations.
        text = re.sub(r'```.*?```', '', text, flags=re.S)
        text = re.sub(r'`[^`\n]+`', '', text)
        for target in re.findall(r'!?\[[^\]\n]*\]\((<[^>]+>|[^)\s]+)(?:\s+"[^"]*")?\)', text):
            target = target.strip('<>')
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            linked = (document.parent / unquote(parsed.path)).resolve()
            try:
                relative = linked.relative_to(root).as_posix()
            except ValueError:
                errors.append(f'{document.relative_to(root)}: link escapes repository: {target}')
                continue
            if not linked.exists() and relative not in known:
                errors.append(f'{document.relative_to(root)}: broken local link: {target}')
    for image in sorted((root / 'docs' / 'assets').glob('*')):
        if image.suffix.lower() == '.svg':
            try:
                element = ET.parse(image).getroot()
                if element.tag != '{http://www.w3.org/2000/svg}svg':
                    raise ValueError('root element is not SVG')
            except (ET.ParseError, ValueError) as exc:
                errors.append(f'{image.relative_to(root)}: {exc}')
        elif image.suffix.lower() == '.png':
            data = image.read_bytes()
            if len(data) < 33 or data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR':
                errors.append(f'{image.relative_to(root)}: invalid PNG header')
            elif 0 in struct.unpack('>II', data[16:24]):
                errors.append(f'{image.relative_to(root)}: empty PNG dimensions')
        elif image.suffix.lower() in ('.jpg', '.jpeg'):
            data = image.read_bytes()
            if not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9'):
                errors.append(f'{image.relative_to(root)}: invalid JPEG markers')
    return errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--inventory', type=Path, help='Optional JSON list of original tracked paths for an offline documentation-only workspace.')
    args = parser.parse_args()
    failures = check(args.root, json.loads(args.inventory.read_text(encoding='utf-8')) if args.inventory else None)
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        sys.exit(1)
    print('Documentation links and image headers are valid.')
