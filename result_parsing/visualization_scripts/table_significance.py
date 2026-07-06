import math
import re
import warnings
from types import SimpleNamespace

try:
    from scipy.stats import ttest_rel
except ImportError:  # pragma: no cover - exercised only in lightweight environments
    ttest_rel = None

SIG_MARKER_PATTERN = r"\$\^\{\\(?:triangle|triangledown|blacktriangle|blacktriangledown)\}\$"
SUMMARY_BODY_PATTERN = r"(?:-|\d+\.\d{3} \(\d+\.\d{3}\)|\\textbf\{(?:-|\d+\.\d{3} \(\d+\.\d{3}\))\})"
SUMMARY_CELL_RE = re.compile(rf"^(?P<body>{SUMMARY_BODY_PATTERN})(?P<marker>{SIG_MARKER_PATTERN})?$")


def get_valid_pairs(x_vals, y_vals):
    return [
        (float(x), float(y))
        for x, y in zip(x_vals, y_vals)
        if math.isfinite(x) and math.isfinite(y)
    ]


def paired_test_pvalue(x_vals, y_vals):
    pairs = get_valid_pairs(x_vals, y_vals)
    if len(pairs) < 2:
        return None

    x = [x for x, _ in pairs]
    y = [y for _, y in pairs]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            if ttest_rel is None:
                result = _fallback_ttest_rel(x, y)
            else:
                try:
                    result = ttest_rel(x, y, alternative="two-sided", nan_policy="omit")
                except TypeError:
                    result = ttest_rel(x, y, nan_policy="omit")
        except TypeError:
            result = _fallback_ttest_rel(x, y)

    p_value = float(result.pvalue)
    if math.isfinite(p_value):
        return p_value

    diffs = [x_val - y_val for x_val, y_val in pairs]
    if all(abs(diff) <= 1e-12 for diff in diffs):
        return 1.0
    if all(abs(diff - diffs[0]) <= 1e-12 for diff in diffs[1:]):
        return 0.0
    return None


def _fallback_ttest_rel(x_vals, y_vals):
    diffs = [float(x) - float(y) for x, y in zip(x_vals, y_vals)]
    n = len(diffs)
    if n < 2:
        return SimpleNamespace(pvalue=float("nan"))

    mean_diff = sum(diffs) / n
    std_diff = sample_std(diffs)
    if std_diff <= 1e-12:
        p_value = 1.0 if abs(mean_diff) <= 1e-12 else 0.0
        return SimpleNamespace(pvalue=p_value)

    t_stat = mean_diff / (std_diff / math.sqrt(n))
    cdf = _student_t_cdf(abs(t_stat), n - 1)
    p_value = max(0.0, min(1.0, 2.0 * (1.0 - cdf)))
    return SimpleNamespace(pvalue=p_value)


def _student_t_pdf(value, degrees_of_freedom):
    numerator = math.gamma((degrees_of_freedom + 1) / 2.0)
    denominator = math.sqrt(degrees_of_freedom * math.pi) * math.gamma(degrees_of_freedom / 2.0)
    return numerator / denominator * (1.0 + (value * value) / degrees_of_freedom) ** (
        -(degrees_of_freedom + 1) / 2.0
    )


def _simpson_integral(fn, left, right):
    mid = (left + right) / 2.0
    return (right - left) * (fn(left) + 4.0 * fn(mid) + fn(right)) / 6.0


def _adaptive_simpson(fn, left, right, eps, whole, depth):
    mid = (left + right) / 2.0
    left_half = _simpson_integral(fn, left, mid)
    right_half = _simpson_integral(fn, mid, right)
    delta = left_half + right_half - whole
    if depth <= 0 or abs(delta) <= 15.0 * eps:
        return left_half + right_half + delta / 15.0
    return _adaptive_simpson(fn, left, mid, eps / 2.0, left_half, depth - 1) + _adaptive_simpson(
        fn,
        mid,
        right,
        eps / 2.0,
        right_half,
        depth - 1,
    )


def _student_t_cdf(value, degrees_of_freedom):
    if value <= 0:
        return 0.5

    fn = lambda point: _student_t_pdf(point, degrees_of_freedom)
    whole = _simpson_integral(fn, 0.0, value)
    area = _adaptive_simpson(fn, 0.0, value, 1e-8, whole, 20)
    return max(0.5, min(1.0, 0.5 + area))


def get_sig_symbol(val, base_val, p):
    if p is None or p >= 0.05:
        return ""
    is_higher = val > base_val
    if p < 0.01:
        return r"$^{\blacktriangle}$" if is_higher else r"$^{\blacktriangledown}$"
    return r"$^{\triangle}$" if is_higher else r"$^{\triangledown}$"


def sample_std(values):
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def format_summary_cell(mean_value, std_value, symbol, bold=False):
    body = f"{mean_value:.3f} ({std_value:.3f})"
    if bold:
        body = rf"\textbf{{{body}}}"
    return f"{body}{symbol}"


def format_summary_with_significance(values, base_values=None, matched_values=None, matched_base_values=None):
    mean_value = sum(values) / len(values)
    std_value = sample_std(values)
    base_mean = sum(base_values) / len(base_values) if base_values else None

    valid_pairs = []
    if matched_values is not None and matched_base_values is not None:
        valid_pairs = get_valid_pairs(matched_values, matched_base_values)

    p_value = paired_test_pvalue(
        [x for x, _ in valid_pairs],
        [y for _, y in valid_pairs],
    )
    symbol = get_sig_symbol(mean_value, base_mean if base_mean is not None else mean_value, p_value)
    return format_summary_cell(mean_value, std_value, symbol), len(valid_pairs), p_value


def merge_significance_markers(existing_text, generated_text):
    existing_lines = existing_text.splitlines()
    generated_lines = generated_text.splitlines()
    if len(existing_lines) != len(generated_lines):
        return generated_text

    merged_lines = []
    for existing_line, generated_line in zip(existing_lines, generated_lines):
        if "&" not in generated_line:
            merged_lines.append(generated_line)
            continue

        existing_cells = existing_line.split("&")
        generated_cells = generated_line.split("&")
        if len(existing_cells) != len(generated_cells):
            merged_lines.append(generated_line)
            continue

        merged_lines.append(
            "&".join(
                _merge_summary_cell(existing_cell, generated_cell)
                for existing_cell, generated_cell in zip(existing_cells, generated_cells)
            )
        )
    return "\n".join(merged_lines)


def _merge_summary_cell(existing_cell, generated_cell):
    leading = generated_cell[: len(generated_cell) - len(generated_cell.lstrip())]
    trailing = generated_cell[len(generated_cell.rstrip()) :]
    generated_core = generated_cell.strip()
    existing_core = existing_cell.strip()

    linebreak = ""
    if generated_core.endswith("\\\\"):
        linebreak = " \\\\"
        generated_core = generated_core[:-2].rstrip()
    if existing_core.endswith("\\\\"):
        existing_core = existing_core[:-2].rstrip()

    existing_match = SUMMARY_CELL_RE.fullmatch(existing_core)
    generated_match = SUMMARY_CELL_RE.fullmatch(generated_core)
    if existing_match and generated_match:
        merged_core = existing_match.group("body") + (generated_match.group("marker") or "")
        return f"{leading}{merged_core}{linebreak}{trailing}"

    return generated_cell
