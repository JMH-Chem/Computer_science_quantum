"""Small reproducible educational examples, not production convergence recommendations."""
from .models import JobSpec


def _atom(symbol, *position):
    return {"symbol": symbol, "position": position}


def examples():
    a = 3.5668
    entries = [
        ("h2", "H₂ · 첫 SCF 계산", "HF/STO-3G의 에너지 수렴과 결합 오비탈을 살펴봅니다.",
         dict(atoms=[_atom("H", 0, 0, 0), _atom("H", 0, 0, .74)])),
        ("water", "H₂O · DFT와 전자밀도", "PBE/def2-SVP. 빠른 예제이며 기저·격자 수렴 검사가 별도로 필요합니다.",
         dict(atoms=[_atom("O", 0, 0, 0), _atom("H", 0, -.757, .587), _atom("H", 0, .757, .587)],
              method="DFT", basis="def2-svp")),
        ("formaldehyde", "포름알데히드 · 흡수와 여기", "TDA/PBE0/def2-SVP. S₁이 항상 가장 강한 흡수선은 아닙니다.",
         dict(atoms=[_atom("C", 0, 0, 0), _atom("O", 0, 0, 1.21), _atom("H", 0, .94, -.59), _atom("H", 0, -.94, -.59)],
              method="DFT", functional="PBE0", task="tda", basis="def2-svp", nstates=4)),
        ("diamond", "다이아몬드 · 결정과 밴드", "PBE/GTH-SZV/GTH-PBE, Γ점·작은 FFT 격자의 빠른 학습 예제입니다.",
         dict(domain="crystal", task="bands", atoms=[_atom("C", 0, 0, 0), _atom("C", a/4, a/4, a/4)],
              lattice=[[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]], method="DFT", basis="gth-szv")),
        ("rhf", "직접 푸는 RHF · 행렬의 의미", "적분은 PySCF, 직교화·Fock 행렬·SCF 반복은 자체 NumPy 구현입니다.",
         dict(atoms=[_atom("O", 0, 0, 0), _atom("H", 0, -.757, .587), _atom("H", 0, .757, .587)],
              task="educational_rhf")),
    ]
    return [{"id": key, "title": title, "description": description,
             "spec": JobSpec(name=title, **spec).model_dump(mode="json")}
            for key, title, description, spec in entries]
