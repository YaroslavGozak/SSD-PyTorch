"""Ukrainian technical report and publication PNG/SVG figures."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .profile_models import ROOT,HERE,RESULTS,COMPONENTS,sha,write_json,write_csv

FIGURES=HERE/'figures'


def main():
    FIGURES.mkdir(exist_ok=True)
    data=json.loads((RESULTS/'results.json').read_text())
    validation=json.loads((RESULTS/'profiler_validation.json').read_text())
    assert validation['status']=='PASS' and validation['production_conv_totals_match_default_THOP']
    assert all(sha(ROOT/p)==value for p,value in data['source_sha256'].items())
    params={r['model']:r for r in data['params']}
    measurements={(r['model'],r['input_h'],r['input_w']):r for r in data['measurements']}
    for r in data['measurements']:
        if r['status']=='PASS':
            r['classification_heads_executed']=sum(e.startswith('cls_') for e in r['execution_events'])
            r['localization_heads_executed']=sum(e.startswith('loc_') for e in r['execution_events'])
            assert r['classification_heads_executed']==r['localization_heads_executed']==r['selected_L_R']
    write_csv('feature_depth_by_shape.csv',[{k:v for k,v in r.items() if k not in ('macs','gmacs')} for r in data['measurements']])
    comp={(r['model'],r['input_h'],r['input_w'],r['component']):r['macs'] for r in data['macs_by_component']}
    ratio_p=params['VGG16']['total_params']/params['MobileNet']['total_params']
    reduction_p=100*(1-1/ratio_p)
    comparisons=[]
    for h,w in sorted({(r['input_h'],r['input_w']) for r in data['measurements']},key=lambda s:(s[0]*s[1],s)):
        a,b=measurements[('VGG16',h,w)],measurements[('MobileNet',h,w)]
        if a['status']!='PASS' or b['status']!='PASS':continue
        comparisons.append({'input_h':h,'input_w':w,'roi_area':h*w,
                            'vgg16_macs':a['macs'],'mobilenet_macs':b['macs'],
                            'vgg16_gmacs':a['gmacs'],'mobilenet_gmacs':b['gmacs'],
                            'ratio':a['macs']/b['macs'],'reduction_percent':100*(1-b['macs']/a['macs']),
                            'vgg16_levels':a['selected_L_R'],'mobilenet_levels':b['selected_L_R']})
    write_csv('macs_comparison.csv',comparisons)
    byshape={(r['input_h'],r['input_w']):r for r in comparisons}
    squares=[r for r in comparisons if r['input_h']==r['input_w']]
    main_squares=[byshape[(s,s)] for s in data['settings']['squares']]
    assert len(main_squares)==11
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','figure.figsize':(8,5)})
    colors={'VGG16':'#2563a6','MobileNet':'#c75d16'}
    def export(name):
        plt.tight_layout()
        for ext in ('png','svg'):plt.savefig(FIGURES/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close()
    plt.figure()
    for model,key in [('VGG16','vgg16_gmacs'),('MobileNet','mobilenet_gmacs')]:
        rows=[r for r in data['measurements'] if r['model']==model and r['status']=='PASS' and r['input_h']==r['input_w']]
        rows.sort(key=lambda r:r['roi_area'])
        plt.plot([r['roi_area'] for r in rows],[r['gmacs'] for r in rows],'.-',color=colors[model],label=model)
        for edge in data['transitions']:
            if edge['model']!=model:continue
            r=measurements[(model,edge['side'],edge['side'])]
            plt.scatter([r['roi_area']],[r['gmacs']],marker='D',color=colors[model],s=28,zorder=4)
            plt.annotate(f'{edge["side"]}:L{edge["new_depth"]}',(r['roi_area'],r['gmacs']),
                         xytext=(4,8 if model=='VGG16' or edge['previous_depth'] is None else -15),textcoords='offset points',fontsize=7,color=colors[model])
    plt.xscale('log');plt.yscale('log');plt.xlabel('Square ROI area, pixels (log scale)')
    plt.ylabel('Executed Conv2d GMACs (log scale)');plt.title('Actual until_1x1 forward; B=1, 31 classes')
    plt.legend();export('macs_vs_area')
    plt.figure()
    plt.plot([r['roi_area'] for r in squares],[r['ratio'] for r in squares],'o-',color='#386c3d',markersize=3)
    plt.xscale('log');plt.xlabel('Square ROI area, pixels (log scale)')
    plt.ylabel('VGG16 / MobileNet Conv2d MAC ratio');plt.title('Whole-architecture computational ratio')
    export('macs_ratio')
    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    palette=['#2563a6','#e59c29','#439b75','#b95062']
    values=np.array([[comp[(model,320,320,c)]/1e9 for c in COMPONENTS[:4]] for model in ('VGG16','MobileNet')])
    totals=values.sum(1)
    for ax,normalized in zip(axes,(False,True)):
        bottom=np.zeros(2)
        for i,component in enumerate(COMPONENTS[:4]):
            v=values[:,i]/totals*100 if normalized else values[:,i]
            ax.bar(['VGG16','MobileNet'],v,bottom=bottom,color=palette[i],label=component)
            if normalized:
                for x,(start,height) in enumerate(zip(bottom,v)):
                    if height>4:ax.text(x,start+height/2,f'{height:.1f}%',ha='center',va='center',fontsize=8,color='white')
            bottom+=v
        ax.set_ylabel('Share, %' if normalized else 'Conv2d GMACs')
        ax.set_title('Component shares' if normalized else 'Absolute convolution MACs')
        if not normalized:
            for i,total in enumerate(totals):ax.text(i,total+.3,f'{total:.3f}',ha='center',fontsize=9)
            ax.set_ylim(0,max(totals)*1.12)
    fig.suptitle('320×320, actual active levels');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,fontsize=8)
    fig.subplots_adjust(bottom=.23,top=.82)
    for ext in ('png','svg'):fig.savefig(FIGURES/f'macs_components.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    primary=byshape[(320,320)]
    pretty=lambda n:f'{n:,}'.replace(',',' ')
    report=['# Порівняльний аналіз Params та MACs ROI-SSD VGG16 і ROI-SSD-MobileNet', '',
            '## 1. Мета експерименту', '',
            'Кількісно порівняти повну кількість параметрів та згорткову складність двох актуальних '
            'детекторів за однакових розмірів ROI й геометричної політики `until_1x1`. Навчання, '
            'датасети, mAP evaluation, latency та FPS не використовувалися.', '',
            '## 2. Архітектури та конфігурації', '',
            '`model/roissd.py:RoiSSD` використовує VGG16, learned L2 scale для conv4_3, '
            'dilated fc6/fc7, додаткові conv8–conv11 та звичайні Conv2d heads. '
            '`model/roissd_mobilenet.py:RoiSSDMobileNet` використовує частину MobileNetV3-Large, '
            'SE-блоки, нові channel adapters і depthwise/pointwise feature layers та heads. '
            'Початкові classifier/невикористані частини torchvision backbone не входять до детекторів.', '',
            'Використано поточні YAML `config/imagenet-vid-roissd.yaml` та '
            '`config/imagenet-vid-roissd-mobilenet.yaml`, 31 вихідний клас (30 foreground + background). '
            'Архітектурні aspect ratios, channels та реально спожиті scales звірено зі збереженими '
            'final inference configs. На локацію припадає 4/6/6/6/4/4 anchors для шести можливих heads. '
            'Історичний VGG YAML уже містить terminal scale 1.0 від старого mutating constructor; '
            'порівнювалися сім реально спожитих scales, а не зайвий невикористаний хвіст списку.', '',
            '**Checkpoint і pretrained weights не завантажувалися.** Factory torchvision тимчасово '
            'отримує `weights=None`; структура і detector forward не змінюються. Всі шари '
            'ініціалізовані до підрахунку; parameter identities не змінювалися при forward. '
            'Ваги випадкові, а MACs визначаються виконаними модулями й формами, не їх значеннями.', '',
            'Структурний поділ компонентів: VGG backbone — усі збережені pretrained feature '
            'convolutions включно з conv5 усередині `conv5_3_fc`; нові dilated fc6/fc7 — additional '
            'feature layers. MobileNet backbone — `features_stage1` та збережений original block '
            '`features_stage2[0]`; додані 112→672 і 160→480 adapters та conv8–11 — additional '
            'layers. Усі depthwise/pointwise/BN всередині detection head належать відповідній head. '
            'VGG `scale_weight` із 512 параметрів виділено окремо. Належність встановлюється '
            'за реальними об’єктами модулів; shared parameters рахуються за унікальним identity.', '',
            '## 3. Методика підрахунку Params/MACs', '',
            'Params — сума `numel()` усіх унікальних параметрів повного детектора, незалежно від '
            'активної глибини. Окремо зафіксовані `requires_grad` та nontrainable parameters. '
            'BN running mean/variance — buffers, а не Params. Поточні BN flags залишають affine '
            'trainable; `eval()` не вимикає `requires_grad`.', '',
            '**1 MAC — одна операція множення-накопичення. Основна метрика — лише щільні Conv2d MACs.** '
            'Для кожного реально виконаного Conv2d використано '
            '`B × Hout × Wout × Cout × (Cin/groups) × Kh × Kw`. Bias additions не включені. '
            'Depthwise та SE 1×1 convolutions враховані цією ж формулою. Це не повна кількість '
            'арифметичних операцій detector forward і не FLOPs; множник 2 не застосовувався.', '',
            f'Правило `dense_conv2d_MAC_v1`; THOP {data["thop"]}, PyTorch {data["torch"]}, '
            f'torchvision {data["torchvision"]}, Python {data["python"]}. CPU, float32, B=1, '
            f'{data["settings"]["threads"]} threads, `model.eval()`, `torch.no_grad()`, synthetic tensors, '
            f'seed={data["settings"]["seed"]}. Git revision `{data["git_revision"]}`; hashes '
            'поточних dirty source/config files записані у `results/results.json`.', '',
            'THOP counting hooks для non-convolution arithmetic занулено в основній метриці. '
            'Щоб не приховати підтримувані ним інші операції, окремо збережено штатний THOP '
            'підрахунок BN/pooling для 320×320 у `thop_supplementary_ops.csv`. Це змішаний '
            'operation count, який не додається до Conv2d MACs і не називається чистими MACs.', '',
            '## 4. Перевірка правильності профілювання', '',
            'Чотири незалежні тести звичайної, grouped, depthwise та depthwise з multiplier=2 '
            'згорток підтвердили формулу MACs і THOP. Для кожного виконаного Conv2d в експерименті '
            f'перевірено рівність аналітичного та THOP increment: {data["conv_hook_analytic_checks_passed"]} '
            'перевірок, повний журнал `conv_layer_validation.csv`. Leaf-level сума збігається '
            'з підсумком; одна convolution не виконувалася повторно в одному forward.', '',
            'Hooks перевіряють порядок feature block→classification head→localization head, '
            'відсутність пізніших блоків після зупинки та `N_loc == N_cls == N_anchors`. '
            'Для 320×320 згорткова частина штатного THOP також збігається з основною метрикою. '
            'Перевірено повторне профілювання після unsupported forward; залишкові THOP hooks '
            'і counters очищуються, тому не накопичуються між ROI.', '',
            'THOP Params порівняно з прямими **active-path** Params. У VGG THOP не враховує '
            'standalone learned L2 scale (512 параметрів); ця різниця пояснена й перевірена. '
            'MobileNet active-path counts збігаються. Основна Params-таблиця завжди показує '
            'повну модель; inactive heads не вилучаються з неї. Значення active/full Params '
            'винесено в різні CSV columns.', '',
            '## 5. Результати Params', '',
            '| Component | ROI-SSD VGG16 | ROI-SSD-MobileNet |','|---|---:|---:|']
    for component in COMPONENTS:
        r={r['model']:r for r in data['params_by_component'] if r['component']==component}
        report.append(f'| {component} | {pretty(r["VGG16"]["params"])} | {pretty(r["MobileNet"]["params"])} |')
    for key,label in [('total_params','Total Params'),('trainable_params','Trainable Params'),('nontrainable_params','Nontrainable Params')]:
        report.append(f'| {label} | {pretty(params["VGG16"][key])} | {pretty(params["MobileNet"][key])} |')
    report+=['',f'`R_P = P_VGG16 / P_MobileNet = {ratio_p:.6f}`; '
             f'`ΔP = (P_VGG16 − P_MobileNet) / P_VGG16 × 100% = {reduction_p:.6f}%`.', '',
             '## 6. MACs для різних ROI', '',
             'Усі запитані основні та rectangular ROI підтримуються обома моделями. '
             'Нижче наведені **Conv2d GMACs**, де 1 GMAC = 10⁹ MACs.', '',
             '| ROI H×W | VGG16 GMACs | MobileNet GMACs | Ratio | Reduction, % | VGG/Mobile levels |',
             '|---|---:|---:|---:|---:|---|']
    requested=[(s,s) for s in data['settings']['squares']]+[tuple(s) for s in data['settings']['rectangles']]
    for shape in requested:
        r=byshape[shape]
        report.append(f'| {shape[0]}×{shape[1]} | {r["vgg16_gmacs"]:.6f} | {r["mobilenet_gmacs"]:.6f} | '
                      f'{r["ratio"]:.3f} | {r["reduction_percent"]:.4f} | {r["vgg16_levels"]}/{r["mobilenet_levels"]} |')
    report+=['','Формули: `R_M = MAC_VGG16 / MAC_MobileNet`; '
             '`ΔM = (MAC_VGG16 − MAC_MobileNet) / MAC_VGG16 × 100%`. '
             'Дані для всіх додаткових та transition ROI збережено у CSV/JSON без екстраполяції за площею.', '',
             '## 7. Вплив динамічної глибини', '',
             'Перебрано кожний цілий square side від 1 до 320 через **реальний feature iterator** '
             'кожної моделі. Для виявлених переходів виконано повний detector forward у точках '
             'n−1/n/n+1, якщо n−1 існує. Розмір 640×640 не використовувався. '
             'VGG16 не підтримує 1–3 px через pooling; мінімальний перевірений валідний square '
             'side — 4. MobileNet підтримав side=1 геометрично; це не твердження про корисну '
             'точність на таких малих входах.', '',
             '| Model | Side n | Попередня→нова глибина | GMACs n−1 | GMACs n | GMACs n+1 |',
             '|---|---:|---|---:|---:|---:|']
    for edge in data['transitions']:
        vals=[]
        for s in (edge['side']-1,edge['side'],edge['side']+1):
            r=measurements.get((edge['model'],s,s),{})
            vals.append(f'{r["gmacs"]:.6f}' if r.get('status')=='PASS' else 'unsupported/—')
        report.append(f'| {edge["model"]} | {edge["side"]} | {edge["previous_depth"] or "—"}→{edge["new_depth"]} | '+' | '.join(vals)+' |')
    report+=['','Політика однакова за принципом, але геометрія різна. VGG має unpadded 3×3 '
             'conv10/conv11 і pooling у conv5/fc. Якщо наступна карта має непозитивний H або W, '
             'forward зупиняється **до невалідної згортки**, навіть коли поточна карта не 1×1. '
             'Наприклад, VGG 128×128 завершується 2×2 з чотирма рівнями. У MobileNet extra '
             'depthwise convolutions мають stride=2/padding=1, і 128×128 завершується 1×1 також '
             'на четвертому рівні. Тому назву until_1x1 не можна трактувати як гарантію '
             'однакової фінальної карти чи однакової глибини для всіх H/W. Усі stop reasons '
             'та feature shapes записані у `feature_depth_by_shape.csv`.', '',
             'На 64×64 VGG/MobileNet мають 4/3 рівні; на 140×140 — 5/5; на 268×268 — 6/6; '
             'на 160×320 — 5/6. Зміна глибини додає head/block, а округлення backbone feature '
             'maps також змінює MACs сусідніх розмірів; величина стрибка не є чистою ціною '
             'однієї нової head.', '',
             '## 8. Складові архітектур і вплив форми', '',
             '| ROI | Model | Backbone GMAC | Additional GMAC | Class GMAC | Loc GMAC | Heads share |',
             '|---|---|---:|---:|---:|---:|---:|']
    for side in (128,256,320):
        for model in ('VGG16','MobileNet'):
            values=[comp[(model,side,side,c)]/1e9 for c in COMPONENTS[:4]]
            share=(values[2]+values[3])/sum(values)*100
            report.append(f'| {side}×{side} | {model} | '+' | '.join(f'{v:.6f}' for v in values)+f' | {share:.2f}% |')
    report+=['','Поділ backbone/adapters не є механічно однаковим за назвами: нові fc6/fc7 VGG '
             'та нові MobileNet adapters віднесено до additional за походженням, а не назвою '
             'контейнера. Інші trainable modules VGG мають 512 Params, але їхня функціональна '
             'арифметика L2/scale не належить до згорткових MACs.', '',
             'За сталої площі 16 384 px VGG має однакові 5.724555 GMAC для 128×128 і 64×256; '
             'MobileNet має відповідно 0.074854 та 0.075087 GMAC, причому активує 4 та 5 рівнів. '
             'Для площі 25 600 px 160×160 і 100×256 також відрізняються через feature geometry '
             'та rounding. У перевірених транспонованих парах H×W/W×H MACs однакові. '
             'Отже, загальна залежність близька до площі, але точні значення не визначаються '
             'лише кількістю пікселів.', '',
             '## 9. Обмеження експерименту', '',
             '1. Основна оцінка охоплює всі виконані Conv2d, а не повну арифметику мережі. '
             'Не враховано bias adds, BN arithmetic, activations (ReLU/ReLU6/Hardswish/Hardsigmoid), '
             'pooling reductions/comparisons, residual adds, SE gating multiplication, VGG L2 '
             'normalization/scale multiplication, softmax, box decoding exp/div/add, anchor generation, '
             'clipping, comparisons/TopK, NMS, preprocessing, copies та memory access. Підтримувані '
             'штатним THOP non-conv operations подані окремо й не змішані з MACs.',
             '2. MACs щільних згорток враховують padded kernel positions за стандартною формулою; '
             'sparsity, operator fusion, hardware utilization та datatype-specific acceleration не оцінено.',
             '3. Випадкові ваги не використовуються для висновків про accuracy. Checkpoint/state '
             'та mAP не оцінювалися. MACs не доводять фактичного прискорення або економії енергії.',
             '4. Повні Params та active-path Params — різні величини. BN buffers не входять до Params. '
             'Результат стосується 31 класу; зміна кількості класів змінить classification heads.', '',
             '## 10. Висновки для конференційних тез', '',
             f'ROI-SSD-MobileNet має у {ratio_p:.3f} раза менше повних параметрів '
             f'({reduction_p:.3f}% reduction). Для 320×320 згорткові MACs знижуються з '
             f'{primary["vgg16_gmacs"]:.6f} до {primary["mobilenet_gmacs"]:.6f} GMAC '
             f'({primary["ratio"]:.3f} раза; {primary["reduction_percent"]:.4f}%). '
             f'Для основних square ROI 64–320 ratio змінюється від {min(r["ratio"] for r in main_squares):.3f} '
             f'до {max(r["ratio"] for r in main_squares):.3f}; відносна економія не є точно сталою. '
             'Для ще менших transition probes відношення змінюється сильніше.', '',
             'Це порівняння **двох цілісних архітектур**, включно з різними additional layers '
             'та heads, а не ізольована оцінка заміни VGG backbone на MobileNet.', '',
             '## Готові результати для наукової публікації', '',
             '| Показник | ROI-SSD VGG16 | ROI-SSD-MobileNet |', '|---|---:|---:|',
             f'| Params, млн | {params["VGG16"]["total_params"]/1e6:.6f} | {params["MobileNet"]["total_params"]/1e6:.6f} |',
             f'| Conv2d GMACs, 128×128 | {byshape[(128,128)]["vgg16_gmacs"]:.6f} | {byshape[(128,128)]["mobilenet_gmacs"]:.6f} |',
             f'| Conv2d GMACs, 256×256 | {byshape[(256,256)]["vgg16_gmacs"]:.6f} | {byshape[(256,256)]["mobilenet_gmacs"]:.6f} |',
             f'| Conv2d GMACs, 320×320 | {primary["vgg16_gmacs"]:.6f} | {primary["mobilenet_gmacs"]:.6f} |',
             '| Активні рівні, 320×320 | 6 | 6 |', '',
             f'Порівняльний аналіз актуальних реалізацій ROI-SSD на основі VGG16 та MobileNetV3 '
             f'зафіксував зменшення кількості параметрів із {params["VGG16"]["total_params"]/1e6:.3f} '
             f'до {params["MobileNet"]["total_params"]/1e6:.3f} млн, що відповідає скороченню '
             f'на {reduction_p:.2f}%. Для входу 320×320 і 31 вихідного класу кількість '
             f'множень-накопичень у фактично виконаних згортках зменшилася з '
             f'{primary["vgg16_gmacs"]:.3f} до {primary["mobilenet_gmacs"]:.3f} GMAC '
             f'({primary["ratio"]:.2f} раза). Обидві моделі використовували геометричний '
             'вибір глибини until_1x1 без відновлення legacy threshold policy.', '',
             'Для сегментів різної форми зафіксовано відмінності кількості активних рівнів '
             'та згорткової складності, зумовлені геометрією feature maps і detection heads. '
             'Отриману економію слід відносити до цілісної архітектури ROI-SSD-MobileNet. '
             'Оцінка охоплює convolutional MACs і не включає іншу арифметику, preprocessing '
             'та NMS; вона не є вимірюванням фактичного прискорення інференсу.', '',
             'Графіки: [MACs vs area PNG](figures/macs_vs_area.png), [SVG](figures/macs_vs_area.svg); '
             '[ratio PNG](figures/macs_ratio.png), [SVG](figures/macs_ratio.svg); '
             '[components PNG](figures/macs_components.png), [SVG](figures/macs_components.svg).', '',
             '![MACs залежно від площі](figures/macs_vs_area.png)', '',
             '![Відношення MACs](figures/macs_ratio.png)', '',
             '![Декомпозиція MACs](figures/macs_components.png)', '',
             'Відтворення з кореня репозиторію:', '', '```powershell',
             '.venv/Scripts/python.exe -m experiments.roi_ssd_params_macs.profile_models',
             '.venv/Scripts/python.exe -m experiments.roi_ssd_params_macs.validate_profiler',
             '.venv/Scripts/python.exe -m experiments.roi_ssd_params_macs.make_report','```']
    with (RESULTS/'thop_supplementary_ops.csv').open(newline='',encoding='utf-8') as stream:
        supplementary=[r for r in csv.DictReader(stream) if r['primary_conv_macs']]
    insertion=report.index('## 4. Перевірка правильності профілювання')
    extra=['Штатний THOP для 320×320 (допоміжний підрахунок, **не чисті MACs**):', '',
           '| Model | Conv2d MACs | Додаткові підтримувані non-conv operations | Штатний змішаний operation count |',
           '|---|---:|---:|---:|']
    for r in supplementary:
        extra.append(f'| {r["model"]} | {int(r["primary_conv_macs"]):,} | '
                     f'{int(r["additional_nonconv_thop_operations"]):,} | {int(r["thop_default_mixed_operation_count"]):,} |')
    extra+=['','Нуль non-conv у VGG не означає відсутності такої арифметики: його функціональна '
            'L2-нормалізація/scale та інші перелічені операції цим профайлером не охоплені. '
            'Додаткова підтримувана арифметика MobileNet наведена для прозорості; жоден із '
            'двох підрахунків не оголошується точною кількістю всіх операцій detector.', '']
    report[insertion:insertion]=extra
    (HERE/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    data.update(status='COMPLETE',validation=validation,
                parameter_ratio=ratio_p,parameter_reduction_percent=reduction_p,
                main_320_comparison=primary,comparison=comparisons)
    for p in ('experiments/roi_ssd_params_macs/validate_profiler.py','experiments/roi_ssd_params_macs/make_report.py'):
        data['source_sha256'][p]=sha(ROOT/p)
    data['output_sha256']={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*')
                          if p.is_file() and '__pycache__' not in str(p) and p.suffix in ('.csv','.png','.svg','.md','.json')
                          and p!=RESULTS/'results.json' and p.name!='config.json'}
    write_json(RESULTS/'results.json',data)
    print(f'COMPLETE: Params {ratio_p:.4f}x; 320x320 conv MACs {primary["ratio"]:.4f}x, reduction {primary["reduction_percent"]:.5f}%')


if __name__=='__main__':main()
