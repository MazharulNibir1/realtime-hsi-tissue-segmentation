from hsi.data.subset import ImageRecord, parse_label_meta, select_images


def _rec(subject, label, situs, angle, rep, ts):
    return ImageRecord(f"{subject}#{ts}", subject, ts, label, situs, angle, rep)


def test_parse_label_meta():
    r = parse_label_meta(
        "P086#t1", {"label_meta": {"spleen": {"situs": 2, "angle": 1, "repetition": 3}}}
    )
    assert (r.subject, r.label, r.situs, r.angle, r.repetition) == ("P086", "spleen", 2, 1, 3)
    two = {
        "label_meta": {
            "a": {"situs": 1, "angle": 0, "repetition": 1},
            "b": {"situs": 1, "angle": 0, "repetition": 1},
        }
    }
    assert parse_label_meta("P086#t2", two) is None


def test_select_one_per_situs_prefers_config():
    recs = [
        _rec("P1", "liver", s, a, r, f"{s}{a}{r}")
        for s in (1, 2, 3, 4)
        for a in (0, 1, 2)
        for r in (1, 2, 3)
    ]
    out = select_images(recs, angle=0, repetition=1, max_per_pair=4)
    assert len(out) == 4
    assert {(r.situs, r.angle, r.repetition) for r in out} == {(s, 0, 1) for s in (1, 2, 3, 4)}


def test_select_falls_back_when_preferred_missing():
    recs = [_rec("P1", "liver", 1, 1, 2, "b"), _rec("P1", "liver", 1, 2, 1, "a")]
    (out,) = select_images(recs, angle=0, repetition=1)
    assert (out.angle, out.repetition) == (1, 2)


def test_select_is_order_independent():
    recs = [
        _rec(p, o, s, 0, 1, f"{p}{o}{s}") for p in ("P1", "P2") for o in ("a", "b") for s in (1, 2)
    ]
    assert select_images(recs) == select_images(list(reversed(recs)))
