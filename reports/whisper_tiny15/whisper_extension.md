# Whisper tiny 15-language extension

The consolidated illustrated analysis is available as [PDF](whisper_tiny15_analysis.pdf)
or [self-contained HTML](whisper_tiny15_analysis.html). It includes all 204 original
plots, nine additional analytical figures, deployment verification and source-level
results. Rebuild the HTML and figures with `uv run python scripts/build_whisper15_report.py`.

One paired seed-42 experiment. Original recording-based splits; speaker independence is not established.
Both variants start from the selected frozen Whisper12 classifier and pretrained tiny audio encoder. Hidden layers and old output rows are copied exactly; three seeded rows are appended.
Whisper fine-tuning updates the complete audio encoder. SpeechBrain partial15 updates its encoder tail.
Selection uses combined VoxLingua/Pacific macro F1, breaking ties by class-balanced NLL. Calibration uses FLEURS validation and Pacific calibration only. External clips never affect fitting, selection or preprocessing.

| Model | Test cohort | Calibration | Group | N | Accuracy | F1 | NLL | Brier | ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| frozen | fleurs_test | joint | all15 | 8519 | 0.9578589036271863 | 0.9268050452557869 | 0.10941797810694892 | 0.05951898325880159 | 0.0069049514217119235 |
| frozen | fleurs_test | joint | old12 | 8519 | 0.9578589036271863 | 0.9268050452557869 | 0.10941797810694892 | 0.05951898325880159 | 0.0069049514217119235 |
| frozen | fleurs_test | old_only | all15 | 8519 | 0.9578589036271863 | 0.9268050452557869 | 0.10850547606393247 | 0.05917255941244436 | 0.0047448431584741985 |
| frozen | fleurs_test | old_only | old12 | 8519 | 0.9578589036271863 | 0.9268050452557869 | 0.10850547606393247 | 0.05917255941244436 | 0.0047448431584741985 |
| frozen | pacific_test | joint | all15 | 4545 | 0.9661166116611661 | 0.971087701107583 | 0.1081248347658281 | 0.05276065912182976 | 0.016746007498079685 |
| frozen | pacific_test | joint | new3 | 4545 | 0.9661166116611661 | 0.971087701107583 | 0.1081248347658281 | 0.05276065912182976 | 0.016746007498079685 |
| frozen | pacific_test | old_only | all15 | 4545 | 0.9661166116611661 | 0.971087701107583 | 0.11532445245532942 | 0.0546005496568001 | 0.02765483090511811 |
| frozen | pacific_test | old_only | new3 | 4545 | 0.9661166116611661 | 0.971087701107583 | 0.11532445245532942 | 0.0546005496568001 | 0.02765483090511811 |
| frozen | combined_test | joint | all15 | 13064 | 0.9607317819963258 | 0.9329060004980025 | 0.1089680901334803 | 0.05716774449559454 | 0.004074928267731325 |
| frozen | combined_test | joint | old12 | 8519 | 0.9578589036271863 | 0.9268050452557869 | 0.10941797810694892 | 0.05951898325880159 | 0.0069049514217119235 |
| frozen | combined_test | joint | new3 | 4545 | 0.9661166116611661 | 0.971087701107583 | 0.1081248347658281 | 0.05276065912182976 | 0.016746007498079685 |
| frozen | combined_test | old_only | all15 | 13064 | 0.9607317819963258 | 0.9329060004980025 | 0.11087781590616295 | 0.05758194517948331 | 0.010374608163157332 |
| frozen | combined_test | old_only | old12 | 8519 | 0.9578589036271863 | 0.9268050452557869 | 0.10850547606393247 | 0.05917255941244436 | 0.0047448431584741985 |
| frozen | combined_test | old_only | new3 | 4545 | 0.9661166116611661 | 0.971087701107583 | 0.11532445245532942 | 0.0546005496568001 | 0.02765483090511811 |
| finetuned | fleurs_test | joint | all15 | 8519 | 0.964432445122667 | 0.9341422329139718 | 0.09275444009044154 | 0.05196588727675207 | 0.005943925455959098 |
| finetuned | fleurs_test | joint | old12 | 8519 | 0.964432445122667 | 0.9341422329139718 | 0.09275444009044154 | 0.05196588727675207 | 0.005943925455959098 |
| finetuned | fleurs_test | old_only | all15 | 8519 | 0.964432445122667 | 0.9341422329139718 | 0.0928031454528807 | 0.05188001242136811 | 0.005075965886193654 |
| finetuned | fleurs_test | old_only | old12 | 8519 | 0.964432445122667 | 0.9341422329139718 | 0.0928031454528807 | 0.05188001242136811 | 0.005075965886193654 |
| finetuned | pacific_test | joint | all15 | 4545 | 0.9973597359735974 | 0.9978302499240108 | 0.009574177108063072 | 0.003753270387200511 | 0.003968782912268875 |
| finetuned | pacific_test | joint | new3 | 4545 | 0.9973597359735974 | 0.9978302499240108 | 0.009574177108063072 | 0.003753270387200511 | 0.003968782912268875 |
| finetuned | pacific_test | old_only | all15 | 4545 | 0.9973597359735974 | 0.9978302499240108 | 0.010267597724326245 | 0.0038363632549393498 | 0.00463010520579841 |
| finetuned | pacific_test | old_only | new3 | 4545 | 0.9973597359735974 | 0.9978302499240108 | 0.010267597724326245 | 0.0038363632549393498 | 0.00463010520579841 |
| finetuned | combined_test | joint | all15 | 13064 | 0.9758879363135333 | 0.9462082720711625 | 0.0638158075693982 | 0.03519259090787486 | 0.003047968546492474 |
| finetuned | combined_test | joint | old12 | 8519 | 0.964432445122667 | 0.9341422329139718 | 0.09275444009044154 | 0.05196588727675207 | 0.005943925455959098 |
| finetuned | combined_test | joint | new3 | 4545 | 0.9973597359735974 | 0.9978302499240108 | 0.009574177108063072 | 0.003753270387200511 | 0.003968782912268875 |
| finetuned | combined_test | old_only | all15 | 13064 | 0.9758879363135333 | 0.9462082720711625 | 0.06408881106630078 | 0.035165500368289526 | 0.0031682188128651697 |
| finetuned | combined_test | old_only | old12 | 8519 | 0.964432445122667 | 0.9341422329139718 | 0.0928031454528807 | 0.05188001242136811 | 0.005075965886193654 |
| finetuned | combined_test | old_only | new3 | 4545 | 0.9973597359735974 | 0.9978302499240108 | 0.010267597724326245 | 0.0038363632549393498 | 0.00463010520579841 |

## SpeechBrain15 test comparison

The archived SpeechBrain test evaluations use the same original FLEURS and recording-held-out Pacific manifests. Both families retain all 15 outputs.

| Model | Group | Calibration | N | Accuracy | F1 | NLL | Brier | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| speechbrain15_frozen | old12 | raw | 8519 | 0.9673670618617208 | 0.9519885827093476 | 0.09965477810405807 | 0.04788630016356601 | 0.010190122521942211 |
| speechbrain15_frozen | old12 | joint_temperature | 8519 | 0.9673670618617208 | 0.9519885827093476 | 0.09361051536341673 | 0.047275553872563 | 0.005030412496516871 |
| speechbrain15_frozen | old12 | old_only_temperature | 8519 | 0.9673670618617208 | 0.9519885827093476 | 0.09336430125075144 | 0.047483352439351714 | 0.005073850747853088 |
| speechbrain15_frozen | new3 | raw | 4545 | 0.9971397139713971 | 0.9974227866875106 | 0.014216521737220133 | 0.005870765939032547 | 0.00707099477060686 |
| speechbrain15_frozen | new3 | joint_temperature | 4545 | 0.9971397139713971 | 0.9974227866875106 | 0.02131584342114877 | 0.0072784123387561964 | 0.01386681713223227 |
| speechbrain15_frozen | all15 | raw | 13064 | 0.9777250459277403 | 0.95912790694532 | 0.06993058373883468 | 0.03326898517194747 | 0.004667341368340695 |
| speechbrain15_frozen | all15 | joint_temperature | 13064 | 0.9777250459277403 | 0.95912790694532 | 0.06845900862906217 | 0.03336044301286062 | 0.0033501849989846563 |
| speechbrain15_partial | old12 | raw | 8519 | 0.977344758774504 | 0.9647022956350818 | 0.07385148190002741 | 0.03517794176261938 | 0.00748657344436204 |
| speechbrain15_partial | old12 | joint_temperature | 8519 | 0.977344758774504 | 0.9647022956350818 | 0.07004110070141653 | 0.03499607078371013 | 0.005555068488903775 |
| speechbrain15_partial | old12 | old_only_temperature | 8519 | 0.977344758774504 | 0.9647022956350818 | 0.0699840816407805 | 0.03520433999368922 | 0.005551346529838685 |
| speechbrain15_partial | new3 | raw | 4545 | 0.9977997799779978 | 0.998155724649751 | 0.010299811530169846 | 0.004326714287302952 | 0.004620329999620701 |
| speechbrain15_partial | new3 | joint_temperature | 4545 | 0.9977997799779978 | 0.998155724649751 | 0.014617275224853813 | 0.005052218495319444 | 0.008615963563952347 |
| speechbrain15_partial | all15 | raw | 13064 | 0.984461114513166 | 0.9702111853778522 | 0.05174168843470264 | 0.024444718486799324 | 0.004133418765506396 |
| speechbrain15_partial | all15 | joint_temperature | 13064 | 0.984461114513166 | 0.9702111853778522 | 0.050759005876632575 | 0.02457852572471322 | 0.0033719987742162234 |
| whisper15_frozen | all15 | raw | 13064 | 0.9607317819963258 | 0.9329060004980025 | 0.11624276182223586 | 0.05879391471890376 | 0.018958550226763553 |
| whisper15_frozen | old12 | raw | 8519 | 0.9578589036271863 | 0.9268050452557869 | 0.11027912614507618 | 0.05938770203642511 | 0.008651704039631669 |
| whisper15_frozen | new3 | raw | 4545 | 0.9661166116611661 | 0.971087701107583 | 0.12742080634010677 | 0.057680939106590376 | 0.0418694616557874 |
| whisper15_frozen | all15 | joint | 13064 | 0.9607317819963258 | 0.9329060004980025 | 0.1089680901334803 | 0.05716774449559454 | 0.004074928267731325 |
| whisper15_frozen | old12 | joint | 8519 | 0.9578589036271863 | 0.9268050452557869 | 0.10941797810694892 | 0.05951898325880159 | 0.0069049514217119235 |
| whisper15_frozen | new3 | joint | 4545 | 0.9661166116611661 | 0.971087701107583 | 0.1081248347658281 | 0.05276065912182976 | 0.016746007498079685 |
| whisper15_frozen | all15 | old_only | 13064 | 0.9607317819963258 | 0.9329060004980025 | 0.11087781590616295 | 0.05758194517948331 | 0.010374608163157332 |
| whisper15_frozen | old12 | old_only | 8519 | 0.9578589036271863 | 0.9268050452557869 | 0.10850547606393247 | 0.05917255941244436 | 0.0047448431584741985 |
| whisper15_frozen | new3 | old_only | 4545 | 0.9661166116611661 | 0.971087701107583 | 0.11532445245532942 | 0.0546005496568001 | 0.02765483090511811 |
| whisper15_finetuned | all15 | raw | 13064 | 0.9758879363135333 | 0.9462082720711625 | 0.07553442008954027 | 0.037283896976235874 | 0.012764548225969282 |
| whisper15_finetuned | old12 | raw | 8519 | 0.964432445122667 | 0.9341422329139718 | 0.11224952881678847 | 0.055377604979662924 | 0.01945277538291492 |
| whisper15_finetuned | new3 | raw | 4545 | 0.9973597359735974 | 0.9978302499240108 | 0.006716815854682746 | 0.0033696398846638307 | 0.000716509205176509 |
| whisper15_finetuned | all15 | joint | 13064 | 0.9758879363135333 | 0.9462082720711625 | 0.0638158075693982 | 0.03519259090787486 | 0.003047968546492474 |
| whisper15_finetuned | old12 | joint | 8519 | 0.964432445122667 | 0.9341422329139718 | 0.09275444009044154 | 0.05196588727675207 | 0.005943925455959098 |
| whisper15_finetuned | new3 | joint | 4545 | 0.9973597359735974 | 0.9978302499240108 | 0.009574177108063072 | 0.003753270387200511 | 0.003968782912268875 |
| whisper15_finetuned | all15 | old_only | 13064 | 0.9758879363135333 | 0.9462082720711625 | 0.06408881106630078 | 0.035165500368289526 | 0.0031682188128651697 |
| whisper15_finetuned | old12 | old_only | 8519 | 0.964432445122667 | 0.9341422329139718 | 0.0928031454528807 | 0.05188001242136811 | 0.005075965886193654 |
| whisper15_finetuned | new3 | old_only | 4545 | 0.9973597359735974 | 0.9978302499240108 | 0.010267597724326245 | 0.0038363632549393498 | 0.00463010520579841 |

## Original-language regression

All 15 outputs remain enabled. Matched old-only temperature calibration is used for comparisons with Whisper12.

```json
[
  {
    "method": "frozen",
    "delta_accuracy": -0.004343232773799777,
    "delta_macro_f1": -0.00189555701157762,
    "delta_nll": 0.011261409622692745,
    "delta_brier": 0.005487259189389131,
    "delta_ece": 0.0002726502861887985
  },
  {
    "method": "finetuned",
    "delta_accuracy": -0.004460617443361903,
    "delta_macro_f1": -0.004787942643509346,
    "delta_nll": 0.011646724220397975,
    "delta_brier": 0.006958248815269456,
    "delta_ece": -1.8128833008889365e-05
  }
]
```

## Updated external validation

Directory labels remain provisional. Domain differences and English code-switching limit interpretation. Twelve-language Pacific predictions are out-of-set diagnostics; Hindi/Thai provide matched recognition comparisons. No winner is selected using external validation.

| Model | Cohort | Policy | N scored | Coverage | Accuracy | F1 | Equal-source accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| speechbrain_original107 | complete | all | 1336 | 1.0 | 0.3106287425149701 | 0.35684426802637664 | 0.21700675448092713 |
| speechbrain_original107 | complete | english_excluded | 1276 | 0.9550898203592815 | 0.32523510971786834 | 0.3893387662471632 | 0.23762564125045138 |
| speechbrain_original107 | complete | shared | 1264 | 0.9461077844311377 | 0.3259493670886076 | 0.3914266648872281 | 0.23948152540281453 |
| speechbrain_original107 | historical | all | 1243 | 1.0 | 0.333869670152856 | 0.3574971936793023 | 0.24800771940677388 |
| speechbrain_original107 | historical | english_excluded | 1183 | 0.9517296862429606 | 0.3508030431107354 | 0.3899961185146849 | 0.2715721614290873 |
| speechbrain_original107 | historical | shared | 1171 | 0.9420756234915527 | 0.3518360375747225 | 0.39208848893888265 | 0.2736931718889309 |
| speechbrain_original107 | added | all | 93 | 1.0 | 0.0 | 0.0 | 0.0 |
| speechbrain_original107 | added | english_excluded | 93 | 1.0 | 0.0 | 0.0 | 0.0 |
| speechbrain_original107 | added | shared | 93 | 1.0 | 0.0 | 0.0 | 0.0 |
| speechbrain_original107 | hindi_thai | all | 485 | 1.0 | 0.8556701030927835 | 0.8998570794933715 | 0.8680270179237085 |
| speechbrain_original107 | hindi_thai | english_excluded | 432 | 0.8907216494845361 | 0.9606481481481481 | 0.9833905844882274 | 0.9505025650018055 |
| speechbrain_original107 | hindi_thai | shared | 425 | 0.8762886597938144 | 0.9694117647058823 | 0.9889267461669506 | 0.9579261016112581 |
| speechbrain_filtered12 | complete | all | 1336 | 1.0 | 0.31137724550898205 | 0.34462519936204145 | 0.21762864502819082 |
| speechbrain_filtered12 | complete | english_excluded | 1257 | 0.9408682634730539 | 0.3309466984884646 | 0.37556602194754585 | 0.24010341595007823 |
| speechbrain_filtered12 | complete | shared | 1250 | 0.9356287425149701 | 0.3304 | 0.3762299613045882 | 0.24010341595007823 |
| speechbrain_filtered12 | historical | all | 1243 | 1.0 | 0.33467417538214 | 0.3469852265497986 | 0.24871845146078952 |
| speechbrain_filtered12 | historical | english_excluded | 1164 | 0.9364440868865648 | 0.35738831615120276 | 0.378149386845039 | 0.27440390394294656 |
| speechbrain_filtered12 | historical | shared | 1157 | 0.9308125502815768 | 0.35695764909248057 | 0.3788384300685584 | 0.27440390394294656 |
| speechbrain_filtered12 | added | all | 93 | 1.0 | 0.0 | 0.0 | 0.0 |
| speechbrain_filtered12 | added | english_excluded | 93 | 1.0 | 0.0 | 0.0 | 0.0 |
| speechbrain_filtered12 | added | shared | 93 | 1.0 | 0.0 | 0.0 | 0.0 |
| speechbrain_filtered12 | hindi_thai | all | 485 | 1.0 | 0.8577319587628865 | 0.9007259686871338 | 0.8705145801127633 |
| speechbrain_filtered12 | hindi_thai | english_excluded | 428 | 0.8824742268041237 | 0.9719626168224299 | 0.9897959183673469 | 0.9604136638003129 |
| speechbrain_filtered12 | hindi_thai | shared | 425 | 0.8762886597938144 | 0.971764705882353 | 0.9897959183673469 | 0.9604136638003129 |
| speechbrain_previous12 | complete | all | 1336 | 1.0 | 0.3008982035928144 | 0.3498176275246339 | 0.2065690017946478 |
| speechbrain_previous12 | complete | english_excluded | 1260 | 0.9431137724550899 | 0.319047619047619 | 0.3847133589302264 | 0.22940898414568556 |
| speechbrain_previous12 | complete | shared | 1254 | 0.938622754491018 | 0.3189792663476874 | 0.384641243545611 | 0.22940898414568556 |
| speechbrain_previous12 | historical | all | 1243 | 1.0 | 0.3234111021721641 | 0.35046268477346476 | 0.23607885919388324 |
| speechbrain_previous12 | historical | english_excluded | 1168 | 0.9396621078037007 | 0.3441780821917808 | 0.38536734422744784 | 0.2621816961664978 |
| speechbrain_previous12 | historical | shared | 1162 | 0.9348350764279968 | 0.3442340791738382 | 0.3852952288428325 | 0.2621816961664978 |
| speechbrain_previous12 | added | all | 93 | 1.0 | 0.0 | 0.0 | 0.0 |
| speechbrain_previous12 | added | english_excluded | 92 | 0.989247311827957 | 0.0 | 0.0 | 0.0 |
| speechbrain_previous12 | added | shared | 92 | 0.989247311827957 | 0.0 | 0.0 | 0.0 |
| speechbrain_previous12 | hindi_thai | all | 485 | 1.0 | 0.8288659793814434 | 0.8872944844447435 | 0.8262760071785914 |
| speechbrain_previous12 | hindi_thai | english_excluded | 427 | 0.8804123711340206 | 0.9414519906323185 | 0.9782608695652174 | 0.9176359365827422 |
| speechbrain_previous12 | hindi_thai | shared | 425 | 0.8762886597938144 | 0.9411764705882353 | 0.9782608695652174 | 0.9176359365827422 |
| speechbrain_frozen15 | complete | all | 1336 | 1.0 | 0.7372754491017964 | 0.7673013024159141 | 0.6049102956698665 |
| speechbrain_frozen15 | complete | english_excluded | 1275 | 0.9543413173652695 | 0.7725490196078432 | 0.8038093199344907 | 0.6288227512764853 |
| speechbrain_frozen15 | complete | shared | 1266 | 0.9476047904191617 | 0.7740916271721959 | 0.8052352437451635 | 0.6295053769393201 |
| speechbrain_frozen15 | historical | all | 1243 | 1.0 | 0.7779565567176187 | 0.7951737284014161 | 0.663705174740108 |
| speechbrain_frozen15 | historical | english_excluded | 1182 | 0.9509251810136766 | 0.8181049069373942 | 0.8317152458417632 | 0.6910336954333864 |
| speechbrain_frozen15 | historical | shared | 1173 | 0.9436846339501207 | 0.8201193520886616 | 0.8339329642954745 | 0.6918138390480548 |
| speechbrain_frozen15 | added | all | 93 | 1.0 | 0.1935483870967742 | 0.32432432432432434 | 0.19334614217817778 |
| speechbrain_frozen15 | added | english_excluded | 93 | 1.0 | 0.1935483870967742 | 0.32432432432432434 | 0.19334614217817778 |
| speechbrain_frozen15 | added | shared | 93 | 1.0 | 0.1935483870967742 | 0.32432432432432434 | 0.19334614217817778 |
| speechbrain_frozen15 | hindi_thai | all | 485 | 1.0 | 0.8206185567010309 | 0.8836420101637492 | 0.81686621092178 |
| speechbrain_frozen15 | hindi_thai | english_excluded | 428 | 0.8824742268041237 | 0.9299065420560748 | 0.9726451701521239 | 0.9053044948867163 |
| speechbrain_frozen15 | hindi_thai | shared | 425 | 0.8762886597938144 | 0.9317647058823529 | 0.9746059544658494 | 0.9081293536437784 |
| speechbrain_partial15 | complete | all | 1336 | 1.0 | 0.7657185628742516 | 0.7892432922186803 | 0.6326799309635901 |
| speechbrain_partial15 | complete | english_excluded | 1274 | 0.9535928143712575 | 0.8029827315541601 | 0.8259448041450618 | 0.6566479177211154 |
| speechbrain_partial15 | complete | shared | 1266 | 0.9476047904191617 | 0.8048973143759873 | 0.8276923252756093 | 0.6588852770219914 |
| speechbrain_partial15 | historical | all | 1243 | 1.0 | 0.8061142397425584 | 0.8193344087063966 | 0.6908207649119894 |
| speechbrain_partial15 | historical | english_excluded | 1181 | 0.9501206757843926 | 0.8484335309060118 | 0.8561032926353833 | 0.7182127497777326 |
| speechbrain_partial15 | historical | shared | 1173 | 0.9436846339501207 | 0.8508098891730606 | 0.8578915311870313 | 0.7207697318358769 |
| speechbrain_partial15 | added | all | 93 | 1.0 | 0.22580645161290322 | 0.3684210526315789 | 0.2256940933247941 |
| speechbrain_partial15 | added | english_excluded | 93 | 1.0 | 0.22580645161290322 | 0.3684210526315789 | 0.2256940933247941 |
| speechbrain_partial15 | added | shared | 93 | 1.0 | 0.22580645161290322 | 0.3684210526315789 | 0.2256940933247941 |
| speechbrain_partial15 | hindi_thai | all | 485 | 1.0 | 0.8309278350515464 | 0.8881997313031795 | 0.838421381497473 |
| speechbrain_partial15 | hindi_thai | english_excluded | 427 | 0.8804123711340206 | 0.9437939110070258 | 0.9791666666666667 | 0.9297571142310859 |
| speechbrain_partial15 | hindi_thai | shared | 425 | 0.8762886597938144 | 0.9435294117647058 | 0.9791666666666667 | 0.9297571142310859 |
| whisper_native | complete | all | 1336 | 1.0 | 0.2844311377245509 | 0.33760068397606086 | 0.20647152891887166 |
| whisper_native | complete | english_excluded | 1227 | 0.9184131736526946 | 0.30969845150774244 | 0.3720468816554226 | 0.22916864801597284 |
| whisper_native | complete | shared | 1222 | 0.9146706586826348 | 0.31014729950900166 | 0.3730197192513369 | 0.23007597969856422 |
| whisper_native | historical | all | 1243 | 1.0 | 0.3057119871279163 | 0.33924533875677454 | 0.23596746162156762 |
| whisper_native | historical | english_excluded | 1138 | 0.915526950925181 | 0.3339191564147627 | 0.37416898276820454 | 0.26190702630396895 |
| whisper_native | historical | shared | 1133 | 0.911504424778761 | 0.33451015004413065 | 0.3751665657177468 | 0.2629439767983591 |
| whisper_native | added | all | 93 | 1.0 | 0.0 | 0.0 | 0.0 |
| whisper_native | added | english_excluded | 89 | 0.956989247311828 | 0.0 | 0.0 | 0.0 |
| whisper_native | added | shared | 89 | 0.956989247311828 | 0.0 | 0.0 | 0.0 |
| whisper_native | hindi_thai | all | 485 | 1.0 | 0.7835051546391752 | 0.861714997122488 | 0.8258861156754866 |
| whisper_native | hindi_thai | english_excluded | 425 | 0.8762886597938144 | 0.8941176470588236 | 0.953988616335847 | 0.9166745920638913 |
| whisper_native | hindi_thai | shared | 422 | 0.8701030927835052 | 0.8981042654028436 | 0.9567347835640518 | 0.9203039187942569 |
| whisper_filtered | complete | all | 1336 | 1.0 | 0.28892215568862273 | 0.33291897749729077 | 0.2107786412017302 |
| whisper_filtered | complete | english_excluded | 1124 | 0.8413173652694611 | 0.34341637010676157 | 0.3658173356602676 | 0.2343419411864356 |
| whisper_filtered | complete | shared | 1120 | 0.8383233532934131 | 0.34375 | 0.36636664901856975 | 0.2350346751765834 |
| whisper_filtered | historical | all | 1243 | 1.0 | 0.31053901850362026 | 0.3348019514711255 | 0.24088987565912023 |
| whisper_filtered | historical | english_excluded | 1057 | 0.8503620273531778 | 0.3651844843897824 | 0.36814349306101074 | 0.26781936135592643 |
| whisper_filtered | historical | shared | 1053 | 0.8471440064360418 | 0.36562203228869894 | 0.3687123469049983 | 0.2686110573446667 |
| whisper_filtered | added | all | 93 | 1.0 | 0.0 | 0.0 | 0.0 |
| whisper_filtered | added | english_excluded | 67 | 0.7204301075268817 | 0.0 | 0.0 | 0.0 |
| whisper_filtered | added | shared | 67 | 0.7204301075268817 | 0.0 | 0.0 | 0.0 |
| whisper_filtered | hindi_thai | all | 485 | 1.0 | 0.7958762886597938 | 0.8676334321719792 | 0.8431145648069208 |
| whisper_filtered | hindi_thai | english_excluded | 423 | 0.8721649484536083 | 0.9125295508274232 | 0.9616428571428571 | 0.9373677647457425 |
| whisper_filtered | hindi_thai | shared | 421 | 0.8680412371134021 | 0.9144893111638955 | 0.96356998264893 | 0.9401387007063335 |
| whisper_frozen | complete | all | 1336 | 1.0 | 0.3031437125748503 | 0.3335699953696558 | 0.21755368500686886 |
| whisper_frozen | complete | english_excluded | 1279 | 0.9573353293413174 | 0.3166536356528538 | 0.3591781660676017 | 0.23837538519456794 |
| whisper_frozen | complete | shared | 1266 | 0.9476047904191617 | 0.3175355450236967 | 0.36005017356553193 | 0.23997401342004546 |
| whisper_frozen | historical | all | 1243 | 1.0 | 0.3258246178600161 | 0.33462533428461877 | 0.24863278286499296 |
| whisper_frozen | historical | english_excluded | 1186 | 0.9541432019308126 | 0.3414839797639123 | 0.36039269406392693 | 0.2724290116509348 |
| whisper_frozen | historical | shared | 1173 | 0.9436846339501207 | 0.34271099744245526 | 0.36128433468831506 | 0.2742560153371948 |
| whisper_frozen | added | all | 93 | 1.0 | 0.0 | 0.0 | 0.0 |
| whisper_frozen | added | english_excluded | 93 | 1.0 | 0.0 | 0.0 | 0.0 |
| whisper_frozen | added | shared | 93 | 1.0 | 0.0 | 0.0 | 0.0 |
| whisper_frozen | hindi_thai | all | 485 | 1.0 | 0.8350515463917526 | 0.8910036818564132 | 0.8702147400274755 |
| whisper_frozen | hindi_thai | english_excluded | 431 | 0.8886597938144329 | 0.9396751740139211 | 0.9753627853330115 | 0.9535015407782718 |
| whisper_frozen | hindi_thai | shared | 425 | 0.8762886597938144 | 0.9458823529411765 | 0.9800693240901213 | 0.9598960536801818 |
| whisper_finetuned | complete | all | 1336 | 1.0 | 0.30763473053892215 | 0.34049388541543785 | 0.21478466632133897 |
| whisper_finetuned | complete | english_excluded | 1269 | 0.9498502994011976 | 0.32387706855791965 | 0.36857964374211044 | 0.23724128974032277 |
| whisper_finetuned | complete | shared | 1260 | 0.9431137724550899 | 0.3238095238095238 | 0.3680516443129206 | 0.23724128974032277 |
| whisper_finetuned | historical | all | 1243 | 1.0 | 0.33065164923572005 | 0.34178814382896017 | 0.24546819008153026 |
| whisper_finetuned | historical | english_excluded | 1177 | 0.9469026548672567 | 0.3491928632115548 | 0.3703873405762872 | 0.27113290256036887 |
| whisper_finetuned | historical | shared | 1168 | 0.9396621078037007 | 0.3493150684931507 | 0.3698919544665252 | 0.27113290256036887 |
| whisper_finetuned | added | all | 93 | 1.0 | 0.0 | 0.0 | 0.0 |
| whisper_finetuned | added | english_excluded | 92 | 0.989247311827957 | 0.0 | 0.0 | 0.0 |
| whisper_finetuned | added | shared | 92 | 0.989247311827957 | 0.0 | 0.0 | 0.0 |
| whisper_finetuned | hindi_thai | all | 485 | 1.0 | 0.8474226804123711 | 0.8963519183164905 | 0.8591386652853559 |
| whisper_finetuned | hindi_thai | english_excluded | 428 | 0.8824742268041237 | 0.9602803738317757 | 0.9854202401372213 | 0.9489651589612911 |
| whisper_finetuned | hindi_thai | shared | 425 | 0.8762886597938144 | 0.96 | 0.9854202401372213 | 0.9489651589612911 |
| speechbrain_finetuned | complete | all | 1336 | 1.0 | 0.3001497005988024 | 0.34767295597484277 | 0.20714257093437416 |
| speechbrain_finetuned | complete | english_excluded | 1267 | 0.9483532934131736 | 0.3164956590370955 | 0.38115935439872695 | 0.22958104685045436 |
| speechbrain_finetuned | complete | shared | 1258 | 0.9416167664670658 | 0.3163751987281399 | 0.3817502568413188 | 0.22958104685045436 |
| speechbrain_finetuned | historical | all | 1243 | 1.0 | 0.32260659694288013 | 0.34863515528755756 | 0.2367343667821419 |
| speechbrain_finetuned | historical | english_excluded | 1175 | 0.9452936444086887 | 0.34127659574468083 | 0.3821348828099497 | 0.26237833925766213 |
| speechbrain_finetuned | historical | shared | 1166 | 0.9380530973451328 | 0.34133790737564323 | 0.38272578525254153 | 0.26237833925766213 |
| speechbrain_finetuned | added | all | 93 | 1.0 | 0.0 | 0.0 | 0.0 |
| speechbrain_finetuned | added | english_excluded | 92 | 0.989247311827957 | 0.0 | 0.0 | 0.0 |
| speechbrain_finetuned | added | shared | 92 | 0.989247311827957 | 0.0 | 0.0 | 0.0 |
| speechbrain_finetuned | hindi_thai | all | 485 | 1.0 | 0.8268041237113402 | 0.8873763972786368 | 0.8285702837374966 |
| speechbrain_finetuned | hindi_thai | english_excluded | 428 | 0.8824742268041237 | 0.9369158878504673 | 0.9764397905759162 | 0.9183241874018174 |
| speechbrain_finetuned | hindi_thai | shared | 425 | 0.8762886597938144 | 0.9364705882352942 | 0.9764397905759162 | 0.9183241874018174 |
| whisper15_frozen | complete | all | 1336 | 1.0 | 0.7694610778443114 | 0.7935635653501975 | 0.6729668239323107 |
| whisper15_frozen | complete | english_excluded | 1280 | 0.9580838323353293 | 0.803125 | 0.8231867038073073 | 0.6946225093557855 |
| whisper15_frozen | complete | shared | 1266 | 0.9476047904191617 | 0.8088467614533965 | 0.8284257954151345 | 0.7003369833581733 |
| whisper15_frozen | historical | all | 1243 | 1.0 | 0.7835880933226066 | 0.7824646276936137 | 0.686428503009431 |
| whisper15_frozen | historical | english_excluded | 1187 | 0.9549477071600966 | 0.8205560235888796 | 0.8121051593333934 | 0.7111778577791166 |
| whisper15_frozen | historical | shared | 1173 | 0.9436846339501207 | 0.8269394714407502 | 0.8173705366934625 | 0.7177086852104172 |
| whisper15_frozen | added | all | 93 | 1.0 | 0.5806451612903226 | 0.7346938775510204 | 0.5787350703924675 |
| whisper15_frozen | added | english_excluded | 93 | 1.0 | 0.5806451612903226 | 0.7346938775510204 | 0.5787350703924675 |
| whisper15_frozen | added | shared | 93 | 1.0 | 0.5806451612903226 | 0.7346938775510204 | 0.5787350703924675 |
| whisper15_frozen | hindi_thai | all | 485 | 1.0 | 0.8371134020618557 | 0.8920740494401951 | 0.8747633698202119 |
| whisper15_frozen | hindi_thai | english_excluded | 433 | 0.8927835051546392 | 0.9376443418013857 | 0.9726639271038056 | 0.9539844360739883 |
| whisper15_frozen | hindi_thai | shared | 425 | 0.8762886597938144 | 0.9458823529411765 | 0.9800693240901213 | 0.9628386577330881 |
| whisper15_finetuned | complete | all | 1336 | 1.0 | 0.6384730538922155 | 0.7455207097746179 | 0.5549519261271206 |
| whisper15_finetuned | complete | english_excluded | 1282 | 0.9595808383233533 | 0.6653666146645866 | 0.7728543284743404 | 0.5762784544760541 |
| whisper15_finetuned | complete | shared | 1266 | 0.9476047904191617 | 0.6690363349131122 | 0.7762483842511338 | 0.5784049096749858 |
| whisper15_finetuned | historical | all | 1243 | 1.0 | 0.670152855993564 | 0.7696053357426942 | 0.6031139473840886 |
| whisper15_finetuned | historical | english_excluded | 1189 | 0.9565567176186646 | 0.7005887300252313 | 0.7972649188713629 | 0.6274871226400128 |
| whisper15_finetuned | historical | shared | 1173 | 0.9436846339501207 | 0.7050298380221653 | 0.8007215416108389 | 0.6299173571530774 |
| whisper15_finetuned | added | all | 93 | 1.0 | 0.21505376344086022 | 0.35398230088495575 | 0.21781777732834462 |
| whisper15_finetuned | added | english_excluded | 93 | 1.0 | 0.21505376344086022 | 0.35398230088495575 | 0.21781777732834462 |
| whisper15_finetuned | added | shared | 93 | 1.0 | 0.21505376344086022 | 0.35398230088495575 | 0.21781777732834462 |
| whisper15_finetuned | hindi_thai | all | 485 | 1.0 | 0.8288659793814434 | 0.8892659473360541 | 0.8343441859775204 |
| whisper15_finetuned | hindi_thai | english_excluded | 432 | 0.8907216494845361 | 0.9305555555555556 | 0.9706926641391347 | 0.9145797526360413 |
| whisper15_finetuned | hindi_thai | shared | 425 | 0.8762886597938144 | 0.9364705882352942 | 0.9764397905759162 | 0.9225162605725492 |

## Samoan volume sensitivity

These probes use checkpoint-selection audio only. Positive gain is capped to avoid clipping; applied gain is recorded in the CSV. They do not alter preprocessing or model selection.

| Model | Requested gain dB | Applied mean dB | N | Changed predictions | Accuracy before | Accuracy after |
|---|---:|---:|---:|---:|---:|---:|
| whisper15_frozen | -12 | -12.0 | 12 | 1 | 0.9166666666666666 | 1.0 |
| whisper15_frozen | 12 | 10.384189804639483 | 12 | 0 | 0.9166666666666666 | 0.9166666666666666 |
| whisper15_finetuned | -12 | -12.0 | 12 | 0 | 1.0 | 1.0 |
| whisper15_finetuned | 12 | 10.384189804639483 | 12 | 1 | 1.0 | 0.9166666666666666 |
| speechbrain_original107 | -12 | -12.0 | 12 | 0 |  |  |
| speechbrain_original107 | 12 | 10.384189804639483 | 12 | 0 |  |  |
| speechbrain_filtered12 | -12 | -12.0 | 12 | 0 |  |  |
| speechbrain_filtered12 | 12 | 10.384189804639483 | 12 | 0 |  |  |
| speechbrain_previous12 | -12 | -12.0 | 12 | 0 |  |  |
| speechbrain_previous12 | 12 | 10.384189804639483 | 12 | 0 |  |  |
| speechbrain_frozen15 | -12 | -12.0 | 12 | 0 | 1.0 | 1.0 |
| speechbrain_frozen15 | 12 | 10.384189804639483 | 12 | 0 | 1.0 | 1.0 |
| speechbrain_partial15 | -12 | -12.0 | 12 | 0 | 1.0 | 1.0 |
| speechbrain_partial15 | 12 | 10.384189804639483 | 12 | 0 | 1.0 | 1.0 |
| whisper_filtered | -12 | -12.0 | 12 | 1 |  |  |
| whisper_filtered | 12 | 10.384189804639483 | 12 | 1 |  |  |
| whisper_frozen | -12 | -12.0 | 12 | 0 |  |  |
| whisper_frozen | 12 | 10.384189804639483 | 12 | 0 |  |  |
| whisper_finetuned | -12 | -12.0 | 12 | 1 |  |  |
| whisper_finetuned | 12 | 10.384189804639483 | 12 | 0 |  |  |
| speechbrain_finetuned | -12 | -12.0 | 12 | 0 |  |  |
| speechbrain_finetuned | 12 | 10.384189804639483 | 12 | 0 |  |  |

## Matched Pacific duration diagnostics

Nested 3/5/10-second excerpts share the same ten-second speech windows. Missing accuracy denotes an out-of-set twelve-language control. Directional confusion rates retain English and other errors in the denominator.

| Model | Language | Seconds | N | Accuracy | English | Niuean to Tongan | Tongan to Niuean |
|---|---|---:|---:|---:|---:|---:|---:|
| whisper15_frozen | sam | 3 | 10 | 0.8 | 0 |  |  |
| whisper15_frozen | sam | 5 | 10 | 0.9 | 0 |  |  |
| whisper15_frozen | sam | 10 | 10 | 0.9 | 0 |  |  |
| whisper15_frozen | ton | 3 | 124 | 0.9435483870967742 | 0 |  | 0.008064516129032258 |
| whisper15_frozen | ton | 5 | 124 | 0.9435483870967742 | 0 |  | 0.016129032258064516 |
| whisper15_frozen | ton | 10 | 124 | 0.9596774193548387 | 0 |  | 0.016129032258064516 |
| whisper15_finetuned | sam | 3 | 10 | 1.0 | 0 |  |  |
| whisper15_finetuned | sam | 5 | 10 | 0.8 | 0 |  |  |
| whisper15_finetuned | sam | 10 | 10 | 0.8 | 0 |  |  |
| whisper15_finetuned | ton | 3 | 124 | 0.6854838709677419 | 0 |  | 0.0967741935483871 |
| whisper15_finetuned | ton | 5 | 124 | 0.6290322580645161 | 0 |  | 0.08064516129032258 |
| whisper15_finetuned | ton | 10 | 124 | 0.7258064516129032 | 1 |  | 0.024193548387096774 |
| speechbrain_original107 | sam | 3 | 10 |  | 0 |  |  |
| speechbrain_original107 | sam | 5 | 10 |  | 0 |  |  |
| speechbrain_original107 | sam | 10 | 10 |  | 0 |  |  |
| speechbrain_original107 | ton | 3 | 124 |  | 0 |  |  |
| speechbrain_original107 | ton | 5 | 124 |  | 0 |  |  |
| speechbrain_original107 | ton | 10 | 124 |  | 0 |  |  |
| speechbrain_filtered12 | sam | 3 | 10 |  | 1 |  |  |
| speechbrain_filtered12 | sam | 5 | 10 |  | 1 |  |  |
| speechbrain_filtered12 | sam | 10 | 10 |  | 0 |  |  |
| speechbrain_filtered12 | ton | 3 | 124 |  | 2 |  |  |
| speechbrain_filtered12 | ton | 5 | 124 |  | 1 |  |  |
| speechbrain_filtered12 | ton | 10 | 124 |  | 0 |  |  |
| speechbrain_previous12 | sam | 3 | 10 |  | 1 |  |  |
| speechbrain_previous12 | sam | 5 | 10 |  | 1 |  |  |
| speechbrain_previous12 | sam | 10 | 10 |  | 0 |  |  |
| speechbrain_previous12 | ton | 3 | 124 |  | 2 |  |  |
| speechbrain_previous12 | ton | 5 | 124 |  | 0 |  |  |
| speechbrain_previous12 | ton | 10 | 124 |  | 0 |  |  |
| speechbrain_frozen15 | sam | 3 | 10 | 0.9 | 0 |  |  |
| speechbrain_frozen15 | sam | 5 | 10 | 0.9 | 0 |  |  |
| speechbrain_frozen15 | sam | 10 | 10 | 0.9 | 0 |  |  |
| speechbrain_frozen15 | ton | 3 | 124 | 0.8064516129032258 | 0 |  | 0.04032258064516129 |
| speechbrain_frozen15 | ton | 5 | 124 | 0.8790322580645161 | 0 |  | 0.04032258064516129 |
| speechbrain_frozen15 | ton | 10 | 124 | 0.9193548387096774 | 0 |  | 0.016129032258064516 |
| speechbrain_partial15 | sam | 3 | 10 | 0.9 | 0 |  |  |
| speechbrain_partial15 | sam | 5 | 10 | 0.9 | 0 |  |  |
| speechbrain_partial15 | sam | 10 | 10 | 0.9 | 0 |  |  |
| speechbrain_partial15 | ton | 3 | 124 | 0.8225806451612904 | 0 |  | 0.03225806451612903 |
| speechbrain_partial15 | ton | 5 | 124 | 0.8870967741935484 | 0 |  | 0.03225806451612903 |
| speechbrain_partial15 | ton | 10 | 124 | 0.9274193548387096 | 0 |  | 0.016129032258064516 |
| whisper_filtered | sam | 3 | 10 |  | 4 |  |  |
| whisper_filtered | sam | 5 | 10 |  | 4 |  |  |
| whisper_filtered | sam | 10 | 10 |  | 3 |  |  |
| whisper_filtered | ton | 3 | 124 |  | 20 |  |  |
| whisper_filtered | ton | 5 | 124 |  | 12 |  |  |
| whisper_filtered | ton | 10 | 124 |  | 6 |  |  |
| whisper_frozen | sam | 3 | 10 |  | 0 |  |  |
| whisper_frozen | sam | 5 | 10 |  | 0 |  |  |
| whisper_frozen | sam | 10 | 10 |  | 0 |  |  |
| whisper_frozen | ton | 3 | 124 |  | 0 |  |  |
| whisper_frozen | ton | 5 | 124 |  | 0 |  |  |
| whisper_frozen | ton | 10 | 124 |  | 0 |  |  |
| whisper_finetuned | sam | 3 | 10 |  | 0 |  |  |
| whisper_finetuned | sam | 5 | 10 |  | 0 |  |  |
| whisper_finetuned | sam | 10 | 10 |  | 0 |  |  |
| whisper_finetuned | ton | 3 | 124 |  | 0 |  |  |
| whisper_finetuned | ton | 5 | 124 |  | 0 |  |  |
| whisper_finetuned | ton | 10 | 124 |  | 2 |  |  |
| speechbrain_finetuned | sam | 3 | 10 |  | 1 |  |  |
| speechbrain_finetuned | sam | 5 | 10 |  | 1 |  |  |
| speechbrain_finetuned | sam | 10 | 10 |  | 0 |  |  |
| speechbrain_finetuned | ton | 3 | 124 |  | 1 |  |  |
| speechbrain_finetuned | ton | 5 | 124 |  | 0 |  |  |
| speechbrain_finetuned | ton | 10 | 124 |  | 0 |  |  |

Detailed predictions, per-language/source tables, confusion matrices, reliability plots (PNG/PDF/SVG), calibration provenance, histories, gain/duration diagnostics and resource measurements are saved under `artifacts/whisper_tiny15` and `artifacts/youtube_validation_v4`.
