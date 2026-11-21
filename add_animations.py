"""Add slide transitions and click-by-click entrance animations to ViT_Presentation.pptx.

build_deck.js names every element that should appear on click n as "a<n>_<k>".
This script turns those names into PowerPoint animations:
  - each click reveals all elements of the next step together (fade in + a short upward float)
  - content slides change with a fade; dark title and section slides with an upward push
Elements without an a<n>_ name (titles, footers) are visible as soon as the slide opens.

    python add_animations.py ViT_Presentation.pptx
"""
import re
import sys
import zipfile
from lxml import etree

NS = {
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}
P = '{%s}' % NS['p']
STEP_NAME = re.compile(r'^a(\d+)_')
EFFECT_MS = 500


def effect(ids, spid, first):
    """One shape's entrance: make visible, fade in, float up 4% of the slide height."""
    i = next(ids)
    node = 'clickEffect' if first else 'withEffect'
    return f'''<p:par><p:cTn id="{i}" presetID="10" presetClass="entr" presetSubtype="0" fill="hold" grpId="0" nodeType="{node}">
<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>
<p:set><p:cBhvr><p:cTn id="{next(ids)}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>
<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl><p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr>
<p:to><p:strVal val="visible"/></p:to></p:set>
<p:animEffect transition="in" filter="fade"><p:cBhvr><p:cTn id="{next(ids)}" dur="{EFFECT_MS}"/>
<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cBhvr></p:animEffect>
<p:anim calcmode="lin" valueType="num"><p:cBhvr additive="base"><p:cTn id="{next(ids)}" dur="{EFFECT_MS}" decel="100000" fill="hold"/>
<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl><p:attrNameLst><p:attrName>ppt_y</p:attrName></p:attrNameLst></p:cBhvr>
<p:tavLst><p:tav tm="0"><p:val><p:strVal val="#ppt_y+0.04"/></p:val></p:tav>
<p:tav tm="100000"><p:val><p:strVal val="#ppt_y"/></p:val></p:tav></p:tavLst></p:anim>
</p:childTnLst></p:cTn></p:par>'''


def timing_xml(steps):
    """steps: {n: [(spid, kind), ...]} -> <p:timing> for a click sequence."""
    counter = iter(range(3, 100000))
    clicks = []
    for n in sorted(steps):
        outer, inner = next(counter), next(counter)
        effects = ''.join(effect(counter, spid, k == 0) for k, (spid, _) in enumerate(steps[n]))
        clicks.append(f'''<p:par><p:cTn id="{outer}" fill="hold"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst><p:childTnLst>
<p:par><p:cTn id="{inner}" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>{effects}</p:childTnLst></p:cTn></p:par>
</p:childTnLst></p:cTn></p:par>''')
    builds = []
    for n in sorted(steps):
        for spid, kind in steps[n]:
            if kind == 'sp':
                builds.append(f'<p:bldP spid="{spid}" grpId="0" animBg="1"/>')
            elif kind == 'chart':
                builds.append(f'<p:bldGraphic spid="{spid}" grpId="0"><p:bldAsOne/></p:bldGraphic>')
    bld = f'<p:bldLst>{"".join(builds)}</p:bldLst>' if builds else ''
    return f'''<p:timing xmlns:p="{NS['p']}"><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>
<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>{"".join(clicks)}</p:childTnLst></p:cTn>
<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>
<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>
</p:childTnLst></p:cTn></p:par></p:tnLst>{bld}</p:timing>'''


def transition_xml(dark):
    inner = '<p:push dir="u"/>' if dark else '<p:fade/>'
    return f'<p:transition xmlns:p="{NS["p"]}" spd="{"slow" if dark else "med"}">{inner}</p:transition>'


def process_slide(xml, dark):
    root = etree.fromstring(xml)
    tree = root.find('p:cSld/p:spTree', NS)
    # make every shape id unique (pptxgenjs can give a table the same id as another shape)
    seen, cnvprs = set(), tree.findall('.//p:cNvPr', NS)
    top = max(int(c.get('id')) for c in cnvprs) if cnvprs else 1
    for c in cnvprs:
        if c.get('id') in seen:
            top += 1
            c.set('id', str(top))
        seen.add(c.get('id'))
    steps = {}
    for el in tree:
        c = el.find('./*/p:cNvPr', NS)
        if c is None:
            continue
        m = STEP_NAME.match(c.get('name', ''))
        if not m:
            continue
        tag = etree.QName(el).localname
        if tag == 'graphicFrame':
            uri = el.find('.//a:graphicData', NS).get('uri', '')
            kind = 'chart' if 'chart' in uri else 'frame'
        else:
            kind = 'sp' if tag == 'sp' else 'other'
        steps.setdefault(int(m.group(1)), []).append((c.get('id'), kind))
    for old in root.findall('p:transition', NS) + root.findall('p:timing', NS):
        root.remove(old)
    anchor = root.find('p:clrMapOvr', NS)
    if anchor is None:
        anchor = root.find('p:cSld', NS)
    pos = list(root).index(anchor) + 1
    root.insert(pos, etree.fromstring(transition_xml(dark)))
    if steps:
        root.insert(pos + 1, etree.fromstring(timing_xml(steps)))
    return etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True), sum(len(v) for v in steps.values()), len(steps)


def main(path):
    zin = zipfile.ZipFile(path)
    files = {n: zin.read(n) for n in zin.namelist()}
    infos = {n: zin.getinfo(n) for n in zin.namelist()}
    zin.close()
    # which layouts are the dark ones (title and section dividers)
    dark_layouts = set()
    for n, data in files.items():
        if re.match(r'ppt/slideLayouts/slideLayout\d+\.xml$', n):
            name = etree.fromstring(data).find('p:cSld', NS).get('name', '')
            if name in ('TITLE_DARK', 'SECTION_DARK'):
                dark_layouts.add(n.split('/')[-1])
    total = 0
    slides = sorted((n for n in files if re.match(r'ppt/slides/slide\d+\.xml$', n)), key=lambda n: int(re.findall(r'\d+', n)[-1]))
    for n in slides:
        rels = files[n.replace('slides/', 'slides/_rels/') + '.rels'].decode()
        layout = re.search(r'slideLayouts/(slideLayout\d+\.xml)', rels).group(1)
        files[n], shapes, clicks = process_slide(files[n], layout in dark_layouts)
        total += clicks
        print(f'{n.split("/")[-1]:14s} {"dark " if layout in dark_layouts else "light"}  {clicks} clicks, {shapes} animated shapes')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for n, data in files.items():
            zout.writestr(infos[n], data)
    print(f'done: {len(slides)} slides, {total} clicks in total')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ViT_Presentation.pptx')
