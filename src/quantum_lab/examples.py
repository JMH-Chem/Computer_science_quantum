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
        ("water_mp2", "H₂O · 전자 상관 MP2", "RHF 기준 에너지와 2차 섭동 상관 에너지를 구분합니다. STO-3G 학습용입니다.",
         dict(atoms=[_atom("O", 0, 0, 0), _atom("H", 0, -.757, .587), _atom("H", 0, .757, .587)],
              method="MP2", basis="sto-3g", threads=1)),
        ("hydrogen_ccsd", "H₂ · CCSD와 상관 에너지", "작은 2전자 계의 CCSD 계산입니다. 저장한 오비탈·밀도는 RHF 기준입니다.",
         dict(atoms=[_atom("H", 0, 0, 0), _atom("H", 0, 0, .74)], method="CCSD", threads=1)),
        ("hydrogen_casscf", "H₂ · CASSCF(2,2)", "2전자·2활성 궤도 계산. 활성 공간 선택에 따라 근사와 비용이 달라집니다.",
         dict(atoms=[_atom("H", 0, 0, 0), _atom("H", 0, 0, 1.5)], method="CASSCF", threads=1)),
        ("water_nevpt2", "H₂O · CASCI 기준 NEVPT2", "정준 RHF의 연속된 2전자·2궤도 공간 위 동적 상관. 이 공간이 연구에 충분하다고 인증하지 않습니다.",
         dict(atoms=[_atom("O", 0, 0, 0), _atom("H", 0, -.757, .587), _atom("H", 0, .757, .587)],
              method="NEVPT2", threads=1)),
        ("water_eom", "H₂O · EOM-EE-CCSD", "CCSD 바닥상태 위 singlet 여기 에너지 3개. 전이 세기와 NTO는 계산하지 않습니다.",
         dict(atoms=[_atom("O", 0, 0, 0), _atom("H", 0, -.757, .587), _atom("H", 0, .757, .587)],
              method="CCSD", task="eomccsd", nstates=3, threads=1)),
        ("water_vibrations", "H₂O · 조화 진동과 IR", "HF/STO-3G의 Hessian과 유한차분 IR. 정지점이 아니면 열화학 값을 제공하지 않습니다. 먼저 최적화할 수 있습니다.",
         dict(atoms=[_atom("O", 0, 0, 0), _atom("H", 0, -.757, .587), _atom("H", 0, .757, .587)],
              task="frequencies", ir_intensities=True, threads=1)),
        ("formaldehyde_tdhf", "포름알데히드 · TDHF", "닫힌껍질 HF의 singlet 선형응답. 여기 에너지는 광학 전이이며 궤도 에너지 간격과 구분합니다.",
         dict(atoms=[_atom("C", 0, 0, 0), _atom("O", 0, 0, 1.21), _atom("H", 0, .94, -.59), _atom("H", 0, -.94, -.59)],
              task="tdhf", basis="sto-3g", nstates=3, threads=1)),
    ]
    return [{"id": key, "title": title, "description": description,
             "spec": JobSpec(name=title, **spec).model_dump(mode="json")}
            for key, title, description, spec in entries]
