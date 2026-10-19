# GraphDRP CCLE 확장 — 전처리 규칙 결정 근거

3개의 애매한 전처리 결정(mutation 판정 기준 / fusion 처리 / CNV 이진화 임계값)에 대해
조사한 근거를 기록합니다. 구현 전 검증용.

## 요약 — 무엇을 어떻게 바꿔서 합쳤는가

GraphDRP는 세포주를 **735차원 이진 벡터**(298개+HLA-A/B 2개 mutation, 10개
fusion, 425개 CNA region)로 표현하고 이 벡터가 학습 입력의 전부입니다
(`graphdrp_adapter.py:63-67`). CCLE의 666개 세포주에 대해 **같은 735개
컬럼, 같은 순서**의 행렬을 새로 만들어서, GDSC로 이미 학습된 모델에 그대로
넣을 수 있게 하는 것이 이번 작업입니다 (재학습 아님, 외부 추론용 입력만
새로 조립).

| # | 구성요소 (735개 중 개수) | GDSC 원본 형식 | CCLE 변환 방법 | 근거/출처 | 코드 위치 | 결과 (coverage) |
|---|---|---|---|---|---|---|
| 1 | Cell 축 정렬 (666개 세포주) | `cosmic_sample_id` | DepMap `ModelID`(ACH-XXXXXX) — 이미 `RNA_combat.csv`가 이 순서로 정렬돼 있어 재매핑 불필요 | 기존 CCLE 파이프라인(`ccle_preprocess.py`)과 동일 키 재사용 | `_canonical_cell_order()` | 666/666, 매핑 이슈 없음 |
| 2 | Mutation (300: 298 driver 유전자 + HLA-A/B) | `<GENE>_mut` = 그 유전자에 coding variant 존재(0/1), `PANCANCER_Genetic_feature.csv` | DepMap `OmicsSomaticMutations.csv`에서 `HugoSymbol`이 패널에 속하고 `VariantInfo`가 non-silent coding consequence(missense/frameshift/stop_gained·lost/start_lost/inframe indel/splice_acceptor·donor/protein_altering)인 행이 1개라도 있으면 1 | 원 정의는 Iorio et al. 2016 STAR Methods §4(COSMIC v68 recurrence filter, PMC4967469)이지만 이건 "패널 선정" 단계일 뿐 — 세포주별 판정은 **MOLI**(Bioinformatics 2019, §2.4.3: "assign ones to genes carrying somatic point mutations") 및 **DeepCDR**(Bioinformatics 2020, §2.5: presence 기반 binary vector) 두 동료심사 선례와 동일한 presence 규칙 채택 | `_build_mutation_matrix()` | 299/300 유전자 resolve(1개 `MLL2`만 ambiguous→NaN), 최종 99.7% |
| 3 | Gene fusion (10) | `<GENE1-GENE2>_mut` | DepMap `OmicsFusionFiltered.csv`의 `LeftGene`/`RightGene` 쌍이 이름과 일치(순서 무관)하면 1. 유전자 심볼은 mutation/CNV와 동일하게 `gene_alias`(HGNC)로 정렬 + GDSC 자체 오타(`EWRS1`→`EWSR1`) 1건 수동 교정 | 문헌 선례 없음 → 사용자 승인 fallback(이름 매칭). `gene_alias` 미적용은 구현 감사에서 버그로 확인되어 수정(아래 "구현 감사" 참고) | `_build_fusion_matrix()` | 8/10 매칭(DepMap 전체 기준), 666셀 내 실값 존재 7/10. `EWSR1-X`(와일드카드)는 사용자 판단으로 폐기하여 항상 NaN. `BCR-ABL`은 `ABL`이 HGNC상 `ABL1`/`DSP-AS1`/`MTTP` 3중 중의적이라 미해결로 보류(사용자 결정) |
| 4 | CNA region (425개 `cnaPANCAN<N>`) | 특정 염색체 구간의 재발성 증폭/결실 여부(0/1) | (a) Iorio 2016 논문 Table S2D에서 425개 리전 전부의 좌표·방향·포함 유전자 추출 → (b) 포함 유전자들의 DepMap `OmicsAbsoluteCNGene.csv` 절대 카피수를 리전 방향에 맞는 임계값과 비교, 하나라도 만족하면 1 | 좌표: Iorio et al. 2016 *Cell* Table S2D(PMC4967469 supplementary mmc3.xlsx, "425/425 ID 완전 일치" 확인 완료). 임계값: PureCN(`OmicsAbsoluteCNGene.csv`를 만든 도구 자신) 공식 함수 `callAlterations()` 기본값 `cutoffs=c(0.5,6,7)` — Riester et al. 2016, *Source Code for Biology and Medicine*, DOI `10.1186/s13029-016-0060-z` | `_build_cnv_matrix()` | 유전자 단위 91.7%(5,135/5,600), 리전 단위(유전자 1개만 살아도 됨) 96.0%(408/425). **한계로 남김(사용자 결정)**: 리전 내 유전자 수와 hit-rate 상관계수 0.53으로 큰 리전이 과대판정되는 경향 확인됨 — "region 전체 신호"가 아니라 "유전자 아무거나 하나"라 구조적 편향 있음 |
| 5 | 컬럼 순서 / 최종 결합 | pivot_table 알파벳순 735열 | 4개 하위 행렬(mutation+fusion+CNV)을 concat 후 GDSC와 동일한 컬럼 순서로 재정렬 | `graphdrp_adapter.py`의 `pivot_table` 결과와 동일 순서 강제 | `main()` | 735/735 컬럼, 최종 결측 19개(2.6%) |

**최종 산출물**: `data/ccle_processed/graphdrp_prep/CCLE_PANCANCER_Genetic_feature_binary.csv`
`[666 cell × 735 feature]`, 전체 coverage 97.4%(716/735 완전 정보, 19개는
결측: mutation 1 + fusion 1(`EWSR1-X`, 의도적 NaN) + CNV 17). Feature
밀도(1의 비율) 3.36% — GDSC 원본 3.94%와 같은 자릿수.

## 구현 감사 (2026-08-21)

엄밀 검사에서 fusion 매칭 함수(`_build_fusion_matrix`)가 mutation/CNV와
달리 `gene_alias` HGNC 해석을 거치지 않고 문자열 그대로 비교하는 버그를
발견 — GDSC가 쓰는 구식 심볼(`ABL`, `MLL`)과 GDSC 자체 오타(`EWRS1`)로
인해 실제로 CCLE에 존재하는 fusion이 "없음"으로 잘못 보고됨. 검증 근거:
- `MLL-AFF1`: DepMap 현재 심볼 `KMT2A`. 666셀 중 `ACH-000782/000874/000130`
  3개에 실제 KMT2A-AFF1 fusion 확인 → 수정 후 3건으로 정상 반영.
- `EWRS1-ERG`(오타, 정확히는 EWSR1-ERG): `ACH-000210`에 실제 EWSR1-ERG
  fusion 확인 → 수정 후 1건으로 정상 반영.
- `BCR-ABL`: `ACH-000326`에 실제 ABL1-BCR fusion 확인되지만, `ABL`이 HGNC
  이력상 3개 유전자(`ABL1`/`DSP-AS1`/`MTTP`)에 중의적으로 쓰여 자동 해석
  거부 대상 — 다른 298개 유전자에 적용한 "모호하면 사람이 판단" 원칙과
  일관되게 **미해결로 유지**(사용자 결정, 2026-08-21).
- CNV의 "region 크기 편향"(위 표 참고)은 방법론적 한계로 판단, 수정하지
  않고 그대로 유지하기로 결정(사용자 결정, 2026-08-21).

---

## 1. Mutation "coding variant 존재" 판정 기준

### GDSC 원본이 실제로 어떻게 만들어졌는가

Iorio et al. 2016, *Cell* 166(3):740-754, "A Landscape of Pharmacogenomic
Interactions in Cancer" — GraphDRP의 `PANCANCER_Genetic_feature.csv`가
유래한 원 논문.

- DOI: `10.1016/j.cell.2016.06.017`
- PMC (오픈 액세스 전문): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4967469/
- STAR Methods 원문 PDF (Europe PMC 경유로 확보):
  https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4967469/supplementaryFiles
  → `mmc1.pdf`, 로컬 사본: `/tmp/iorio_supp/mmc1.pdf` (섹션 "4. Variants
  recurrence filter")

**원문 인용** (mmc1.pdf, STAR Methods §4):

> "Variants from both the cell lines and tumors were screened against the
> 'systematic screen data' from COSMIC (v68) ... to identify the recurrent
> variants most likely to contribute to carcinogenesis ('driver mutations').
> ... For missense variants the number of non-synonymous variants in each
> codon of all genes within the systematic screen data in COSMIC (v68) was
> calculated, and any codon with ≥ 3 variants was classed as recurrently
> mutated. Inframe indels were treated in the same manner ... With regards to
> protein truncating variants all genes that contained > 10 truncating
> variants ... within the systematic screen data were classed as recurrently
> inactivated."

**해석**: 이 재발성 필터(recurrence filter)는 **"어떤 유전자/코돈이 298개
feature 패널에 들어갈 자격이 있는가"를 정하는 1회성 selection 단계**였습니다.
그 selection의 산출물이 이미 우리가 갖고 있는 298개 유전자 목록 그 자체입니다
(재계산 불필요). **패널이 정해진 뒤, 개별 세포주에 0/1 값을 매기는 단계는
별개 문제**이며, 이 단계의 규칙은 원문 어디에도 "코돈별 재계산을 세포주마다
반복하라"고 쓰여있지 않습니다 — 실무적으로는 "그 유전자에 non-silent 변이가
있으면 1"이라는 단순 presence 규칙일 개연성이 높습니다 (원문이 이 마지막
단계를 명시적으로 서술하진 않음 — 아래 문헌 선례로 보강).

### 선례 1 — MOLI (동일 문제를 실제로 풀었던 논문)

Sharifi-Noghabi et al., "MOLI: multi-omics late integration with deep neural
networks for drug response prediction", *Bioinformatics* 35(14):i501–i509, 2019.
**GDSC로 학습 → TCGA/PDX로 외부검증** — 지금 우리 상황과 동일한 구조.

- 논문: https://academic.oup.com/bioinformatics/article/35/14/i501/5529255
- DOI: `10.1093/bioinformatics/btz318`
- bioRxiv preprint: https://www.biorxiv.org/content/10.1101/531327v1.full

**원문 인용** (§2.4.3, Somatic point mutations):
> "Similarly with previous works, we assign ones to genes carrying somatic
> point mutations and zeros to all others."

**원문 인용** (§2.4.2, Copy number aberrations):
> "...binarize gene-level copy number estimates assigning zeros to
> copy-neutral genes and ones to all genes overlapping deletions or
> amplifications."

→ 유전자 단위 **presence/absence** 이진화. 코호트 간 이식 시 코돈-재발성
통계를 다시 계산하지 않음. "Similarly with previous works"라는 문구로 보아
이 자체가 이미 관행이었음을 시사.

### 선례 2 — DeepCDR (고정 유전자 패널 + presence 인코딩)

Liu et al., "DeepCDR: a hybrid graph convolutional network for predicting
cancer drug response", *Bioinformatics* 36(Suppl_2):i911–i918, 2020.

- 논문: https://academic.oup.com/bioinformatics/article/36/Supplement_2/i911/6055929
- DOI: `10.1093/bioinformatics/btaa822`
- GitHub: https://github.com/kimmo1019/DeepCDR

**원문 인용** (§2.5, Data preparation):
> "we only consider data related to 697 genes from COSMIC Cancer Gene Census
> (https://cancer.sanger.ac.uk/census). For genomic mutation data, 34,673
> unique mutation positions including SNPs and Indels within the above genes
> were collected. The genomic mutation of each cancer cell line was
> represented as a binary feature vector in which '1' denotes a mutated
> position and '0' denotes a non-mutated position."

→ 유전자 패널은 **고정된 재사용 가능 리소스**(COSMIC Cancer Gene Census)에서
가져오고, 개별 세포주 값은 **presence 기반**.

### 채택 규칙

298개 고정 유전자 패널(이미 확보됨) × CCLE 세포주 → **그 유전자에 non-silent
coding variant(missense / frameshift / stop_gained·lost / start_lost /
inframe indel / splice_acceptor·donor / protein_altering)가 하나라도 있으면
1, 없으면 0.** MOLI·DeepCDR 두 선례와 정확히 같은 방식.

---

## 2. Fusion feature 12개 (BCR-ABL 등)

문헌에서 cross-cohort fusion feature 재현 선례를 찾지 못함 (검색 결과 전부
발현량/mutation/CNV 통합에 집중, fusion 전용 cross-cohort 매칭 방법론 없음).

**채택 규칙** (선례 없어 사용자 지정 fallback): DepMap `OmicsFusionFiltered.csv`의
`LeftGene`/`RightGene` 쌍이 GDSC fusion 이름의 구성 유전자와 일치하면 1.
매칭 실패분은 결측(0/NaN) 처리.

---

## 3. CNV 이진화 임계값

GDSC 원본은 GISTIC2 자체 알고리즘(원 논문 STAR Methods §5, TCGA 세그먼트
데이터에 ADMIRE 알고리즘 적용)을 쓰는데, CCLE 쪽 DepMap `OmicsAbsoluteCNGene.csv`는
완전히 다른 파이프라인(GATK CNV + PureCN)이라 동일 재현이 원천적으로 불가능.

### 근거 — PureCN 자체 공식 기본값

`OmicsAbsoluteCNGene.csv`는 DepMap의 임의 후처리가 아니라 **PureCN이라는
특정 도구가 직접 산출한 절대 카피수**입니다 (DepMap README.txt 1066-1070행:
"Gene-level absolute copy number data generated from PureCN"). 따라서
가장 근거가 강한 임계값은 그 데이터를 만든 도구 자신의 공식 기본 설정입니다.

- PureCN 원 논문: Riester et al., "PureCN: copy number calling and SNV
  classification using targeted short read sequencing", *Source Code for
  Biology and Medicine* 11:13, 2016.
  DOI: `10.1186/s13029-016-0060-z` / PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC5157099/
- 공식 Bioconductor 함수 문서 (`callAlterations`):
  https://rdrr.io/bioc/PureCN/man/callAlterations.html
  / GitHub 소스: https://github.com/lima1/PureCN

**인용** (`callAlterations` 함수 기본 인자):
> `cutoffs = c(0.5, 6, 7)` — 절대 카피수 기준: 0.5 미만은 loss(결실),
> focal amplification은 6 이상, broad(non-focal) amplification은 7 이상.

즉 diploid=2 스케일에서 **deletion: CN < 0.5**, **amplification: CN ≥ 6
(focal) / CN ≥ 7 (broad)** — PureCN을 쓰는 모든 파이프라인의 사실상 표준
기본값입니다.

**주의**: 이것도 "일반적 기본 설정"이지 GDSC가 원래 쓴 GISTIC2/ADMIRE
파이프라인의 공식 대체재는 아닙니다 — 완전히 다른 두 알고리즘이라 100% 동일
재현은 원천적으로 불가능하다는 한계는 여전합니다. 다만 **데이터를 실제로 만든
도구 자신의 공식 기본값**이라는 점에서, 근거 없이 임의로 고른 숫자보다는
훨씬 방어 가능한 선택입니다.

**채택 규칙**: 리전(`cnaPANCAN<N>`)의 지정 방향이 Amplification이면 리전 내
유전자 중 하나라도 **CN ≥ 6(focal) 또는 CN ≥ 7(broad)** → 1. Deletion이면
하나라도 **CN < 0.5** → 1. (PureCN 공식 기본값 채택 — 이전 버전의 CN>3/CN<0.2
초안보다 이 값을 우선한다.)

---

## 아직 확보 못한 것 / 리스크

- Fusion 12개: 이름 매칭 실패율이 얼마나 될지 실제 돌려보기 전엔 모름.
- CNV 임계값(PureCN 기본값)도 GISTIC2/ADMIRE의 공식 대체재는 아님 — 최종
  리포트에 "근사치"임을 명시해야 함.
- Mutation 규칙은 MOLI/DeepCDR 선례와 일치하지만, GDSC 원 논문이 이 마지막
  단계(패널 확정 후 세포주별 판정)를 명시적으로 서술하지 않아 100% 확정은
  아님 — 정황 증거 기반 채택.

---

## 실행 계획

세 가지 규칙 모두 근거가 확보됐으므로(mutation: MOLI/DeepCDR 선례, CNV: PureCN
공식 기본값, fusion: 사용자 승인 fallback), 실제 매트릭스 조립으로 진행.

### Phase 1 — 세 행렬 조립 (`omicsdrp/benchmark/graphdrp_ccle_preprocess.py` 신규 작성)

1. **Mutation 행렬** `[666 cell × 298 gene]`: `OmicsSomaticMutations.csv`에서
   `HugoSymbol`이 298개 패널에 속하고 `VariantInfo`가 non-silent coding
   consequence(missense/frameshift/stop_gained·lost/start_lost/inframe
   indel/splice_acceptor·donor/protein_altering)인 행이 하나라도 있으면 1.
   유전자 심볼은 기존 `gene_alias.py`로 정렬.
2. **Fusion 행렬** `[666 cell × 12 fusion]`: `OmicsFusionFiltered.csv`의
   `LeftGene`/`RightGene` 쌍이 GDSC fusion 이름의 구성 유전자와 일치하면 1.
   매칭 실패는 0/NaN.
3. **CNV 행렬** `[666 cell × 425 region]`: `gdsc_cnaPANCAN_regions.csv`의
   리전별 포함 유전자를 `gene_alias.py`로 CCLE 컬럼에 매핑 후,
   `OmicsAbsoluteCNGene.csv`에서 해당 유전자들의 CN을 조회 — 리전 방향이
   Amplification이면 CN≥6(focal)/≥7(broad) 중 하나라도 만족 시 1, Deletion이면
   CN<0.5 하나라도 만족 시 1.
4. 세 행렬을 `[666 × 735]`로 concat, 컬럼 순서를 GDSC 학습 때 pivot 순서
   (알파벳순, `graphdrp_adapter.py`의 `pivot_table` 결과)와 동일하게 맞춤.
5. 커버리지 리포트 출력: 유전자/리전별 gene_alias 해석 성공률, fusion 매칭
   성공률, 최종 735개 feature 중 완전 결측 비율.

### Phase 2 — 검증

- GDSC 자체 데이터로 같은 파이프라인을 역으로 돌려서(같은 코드로 GDSC
  cosmic_sample_id 기준 재계산) 원본 `PANCANCER_Genetic_feature.csv`와
  일치율을 비교 — 코드 자체의 버그를 잡는 sanity check (CCLE의 "정답"은
  당연히 알 수 없지만, 최소한 파이프라인 로직이 GDSC에서 원본과 얼마나
  가까운 값을 재현하는지는 확인 가능).
- 결측 비율이 지나치게 높은 feature(예: 유전자 매핑 실패로 항상 0인 컬럼)는
  따로 로깅해서 이후 결과 해석 시 "이 feature는 신뢰도 낮음"으로 표시.

### Phase 3 — 추론 연결

- 다른 3개 모델(omicsdrp/deeptta/paccmann)의 `ccle_infer_*.py` 패턴을 따라
  `ccle_infer_graphdrp.py` 작성 — GDSC로 이미 학습된 GraphDRP 모델(4개
  encoder 전부 포함 가능)에 위 `[666×735]` 행렬 + CCLE 약물 SMILES를 넣어
  추론, `BenchmarkResults/ccle_external/`에 다른 모델과 같은 포맷으로 저장.
- `results_summary.ipynb`의 "External Benchmark Table" 셀은 이미
  `ccle_external/*` 파일을 자동 스캔하므로 코드 수정 없이 재실행만 하면 표에
  반영됨(앞서 GraphDRP 4-encoder 벤치마크 때와 동일 패턴).

### 순서

Phase 1(조립 스크립트 작성 + 실행) → 커버리지 확인 후 여기서 한 번 더
컨펌 요청 → Phase 2(검증) → Phase 3(추론 연결 + 노트북 반영).
