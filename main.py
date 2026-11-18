"""Mini NPU Simulator — Python 3.8+, standard library only."""
import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from time import perf_counter_ns

EPSILON = 1e-9
ROOT = Path(__file__).resolve().parent


def normalize_label(label):
    if not isinstance(label, str):
        raise ValueError("라벨은 문자열이어야 합니다.")
    labels = {"+": "Cross", "cross": "Cross", "x": "X"}
    try:
        return labels[label.strip().lower()]
    except KeyError:
        raise ValueError("지원하지 않는 라벨: {!r}".format(label))


def validate_matrix(value, size):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError("행 개수 오류: {}행이 필요합니다.".format(size))
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("열 개수 오류: 각 행에 {}개가 필요합니다.".format(size))
        converted = []
        for number in row:
            if type(number) not in (int, float):
                raise ValueError("행렬 값은 숫자여야 합니다. (bool 제외)")
            try:
                number = float(number)
            except (OverflowError, ValueError):
                raise ValueError("표현 가능한 유한한 숫자가 필요합니다.")
            if not math.isfinite(number):
                raise ValueError("NaN/무한대는 사용할 수 없습니다.")
            converted.append(number)
        result.append(converted)
    return result


def generate_patterns(size):
    if type(size) is not int or size < 3 or size % 2 == 0:
        raise ValueError("크기는 3 이상의 홀수여야 합니다.")
    middle = size // 2
    cross = [[float(r == middle or c == middle) for c in range(size)] for r in range(size)]
    x = [[float(r == c or r + c == size - 1) for c in range(size)] for r in range(size)]
    return cross, x


def mac_2d(pattern, kernel):
    total = 0.0
    for r in range(len(pattern)):
        for c in range(len(pattern)):
            total += pattern[r][c] * kernel[r][c]
    return total


def flatten(matrix):
    return [value for row in matrix for value in row]


def mac_1d(pattern, kernel):
    total = 0.0
    for i in range(len(pattern)):
        total += pattern[i] * kernel[i]
    return total


def decide(a, b, labels=("Cross", "X")):
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("MAC 결과가 유한하지 않습니다. 입력값의 크기를 확인하세요.")
    if abs(a - b) < EPSILON:
        return "UNDECIDED"
    return labels[0] if a > b else labels[1]


def benchmark_row(size, pattern, kernel, repeats):
    # Flattening and I/O are outside the measured function-call interval.
    flat_pattern, flat_kernel = flatten(pattern), flatten(kernel)
    mac_2d(pattern, kernel)
    mac_1d(flat_pattern, flat_kernel)
    samples_2d, samples_1d = [], []
    for i in range(repeats):
        order = [(mac_2d, pattern, kernel, samples_2d),
                 (mac_1d, flat_pattern, flat_kernel, samples_1d)]
        if i % 2:
            order.reverse()
        for function, a, b, samples in order:
            start = perf_counter_ns()
            function(a, b)
            samples.append((perf_counter_ns() - start) / 1_000_000)
    return {"size": size, "operations": size * size,
            "mean_2d_ms": mean(samples_2d), "mean_1d_ms": mean(samples_1d),
            "median_2d_ms": median(samples_2d), "median_1d_ms": median(samples_1d),
            "repeats": repeats}


def performance(sizes, repeats, actual=None):
    print("\n[성능 분석] 한 필터 MAC 함수 호출 평균 / {}회".format(repeats))
    print("크기       2D 평균(ms)    1D 평균(ms)    MAC 횟수(N²)    2D/1D")
    rows = []
    for size in sizes:
        pattern, kernel = actual if actual is not None else generate_patterns(size)
        row = benchmark_row(size, pattern, kernel, repeats)
        rows.append(row)
        ratio = row["mean_2d_ms"] / row["mean_1d_ms"] if row["mean_1d_ms"] else 0
        print("{0:2}x{0:<2} {1:14.6f} {2:14.6f} {3:14} {4:9.3f}".format(
            size, row["mean_2d_ms"], row["mean_1d_ms"], size * size, ratio))
    print("1D 변환 시간 제외. 실제 속도 향상은 환경에 따라 달라질 수 있습니다.")
    return rows


def read_integer(prompt, minimum, maximum=None):
    while True:
        try:
            value = int(input(prompt).strip())
            if value < minimum or (maximum is not None and value > maximum):
                raise ValueError()
            return value
        except ValueError:
            print("입력 형식 오류: 허용 범위의 정수를 입력하세요.")


def read_matrix(name):
    print("{} (3줄 입력, 각 줄 숫자 3개, 공백 구분)".format(name))
    result = []
    while len(result) < 3:
        parts = input("{}행: ".format(len(result) + 1)).split()
        try:
            if len(parts) != 3:
                raise ValueError()
            row = [float(value) for value in parts]
            if not all(math.isfinite(value) for value in row):
                raise ValueError()
            result.append(row)
        except ValueError:
            print("입력 형식 오류: 각 줄에 3개의 유한한 숫자를 공백으로 구분해 입력하세요.")
    return result


def manual_mode(repeats):
    a = read_matrix("필터 A")
    b = read_matrix("필터 B")
    print("필터 A, B 저장 완료")
    for name, matrix in [("A", a), ("B", b)]:
        print(name, matrix)
    print("패턴 입력: 1. 직접 입력  2. 자동 생성 패턴 사용")
    choice = read_integer("선택: ", 1, 2)
    if choice == 1:
        pattern = read_matrix("패턴")
    else:
        cross, x = generate_patterns(3)
        pattern = (cross, x)[read_integer("Cross=1 / X=2: ", 1, 2) - 1]
        print("생성 패턴:", pattern)
    score_a, score_b = mac_2d(pattern, a), mac_2d(pattern, b)
    prediction = decide(score_a, score_b, ("A", "B"))
    print("A 점수: {:.16g}\nB 점수: {:.16g}".format(score_a, score_b))
    print("판정:", "판정 불가 (UNDECIDED)" if prediction == "UNDECIDED" else prediction)
    performance([3], repeats, (pattern, a))


def load_filters(data):
    filters = data.get("filters")
    if not isinstance(filters, dict):
        raise ValueError("filters는 객체여야 합니다.")
    loaded, errors = {}, {}
    for size in (5, 13, 25):
        key = "size_{}".format(size)
        try:
            item = filters.get(key)
            if not isinstance(item, dict):
                raise ValueError("필터 객체 누락/형식 오류")
            normalized = {}
            for label, matrix in item.items():
                label = normalize_label(label)
                if label in normalized:
                    raise ValueError("중복 정규화 라벨: " + label)
                normalized[label] = validate_matrix(matrix, size)
            if set(normalized) != {"Cross", "X"}:
                raise ValueError("Cross, X 필터가 모두 필요합니다.")
            loaded[size] = normalized
            print(key, "필터 로드 완료 (Cross, X)")
        except (ValueError, TypeError) as error:
            errors[size] = str(error)
            print(key, "필터 오류:", error)
    return loaded, errors


def analyze(path, repeats):
    try:
        with path.open(encoding="utf-8-sig") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("최상위 JSON은 객체여야 합니다.")
        loaded, errors = load_filters(data)
        patterns = data.get("patterns")
        if not isinstance(patterns, dict) or not patterns:
            raise ValueError("patterns는 비어 있지 않은 객체여야 합니다.")
    except (OSError, UnicodeError, ValueError) as error:
        print("데이터 로드 실패:", error)
        return 1
    results = []
    for key, item in patterns.items():
        record = {"id": key, "passed": False}
        try:
            match = re.fullmatch(r"size_(\d+)_(\d+)", key)
            if not match:
                raise ValueError("패턴 키는 size_N_idx 형식이어야 합니다.")
            size = int(match.group(1))
            if size not in loaded:
                raise ValueError("필터 사용 불가: " + errors.get(size, "지원하지 않는 크기"))
            if not isinstance(item, dict):
                raise ValueError("패턴 항목은 객체여야 합니다.")
            pattern = validate_matrix(item.get("input"), size)
            expected = normalize_label(item.get("expected"))
            a = mac_2d(pattern, loaded[size]["Cross"])
            b = mac_2d(pattern, loaded[size]["X"])
            prediction = decide(a, b)
            passed = prediction == expected
            reason = "" if passed else ("epsilon 동점 규칙" if prediction == "UNDECIDED" else "expected와 판정 불일치")
            record.update(score_cross=a, score_x=b, prediction=prediction,
                          expected=expected, passed=passed, reason=reason)
            print("\n{} | Cross={:.16g} X={:.16g} | 판정={} expected={} | {}".format(
                key, a, b, prediction, expected, "PASS" if passed else "FAIL: " + reason))
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            record["reason"] = "데이터/스키마 오류: " + str(error)
            print("\n{} | FAIL: {}".format(key, record["reason"]))
        results.append(record)
    rows = performance([3, 5, 13, 25], repeats)
    passed = sum(record["passed"] for record in results)
    print("\n[결과 요약] 총 테스트: {} / 통과: {} / 실패: {}".format(len(results), passed, len(results) - passed))
    for record in results:
        if not record["passed"]:
            print("- {}: {}".format(record["id"], record["reason"]))
    report = {"source": str(path), "epsilon": EPSILON, "total": len(results),
              "passed": passed, "failed": len(results) - passed,
              "cases": results, "performance": rows, "filter_errors": errors}
    try:
        output = ROOT / "reports" / (path.stem + "-report.json")
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print("리포트 저장:", output)
    except OSError as error:
        print("리포트 저장 실패:", error)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="JSON 파일 경로; 지정 시 모드 2 바로 실행")
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--generate", type=int, help="홀수 N 크기 Cross/X를 JSON으로 출력")
    args = parser.parse_args()
    if args.repeats < 10:
        parser.error("반복 횟수는 최소 10회입니다.")
    if args.generate is not None:
        try:
            cross, x = generate_patterns(args.generate)
        except ValueError as error:
            parser.error(str(error))
        print(json.dumps({"cross": cross, "x": x}, ensure_ascii=False))
        return 0
    if args.data is not None:
        return analyze(args.data.resolve(), args.repeats)
    print("=== Mini NPU Simulator ===\n1. 사용자 입력 (3x3)\n2. data.json 분석")
    if read_integer("선택: ", 1, 2) == 1:
        manual_mode(args.repeats)
        return 0
    return analyze(ROOT / "data.json", args.repeats)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n입력이 중단되어 종료합니다.")
