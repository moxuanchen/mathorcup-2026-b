"""Export evidence-bound paper values with ASCII-only JSON keys.

Only the two 05 JSON artifacts are written. Original result files are read-only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / '求解结果' / '数据'
VALUES_PATH = DATA / '05_paper_values.json'
PROVENANCE_PATH = DATA / '05_value_provenance.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_number(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    sources, inventory, provenance = {}, [], {}
    for path in sorted(DATA.rglob('*')):
        if 'cache' in path.parts or not path.is_file():
            continue
        if path.parent == DATA and not path.name.startswith(('00', '01', '02', '03', '04', '06')):
            continue
        if path.suffix.lower() not in ('.json', '.csv'):
            continue
        rel = path.relative_to(ROOT).as_posix()
        record = {'source_file': rel, 'sha256': sha(path), 'bytes': path.stat().st_size}
        if path.suffix.lower() == '.json':
            sources[path.relative_to(DATA).as_posix()] = json.loads(path.read_text(encoding='utf-8-sig'))
            record['format'] = 'json'
        else:
            with path.open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                headers = reader.fieldnames or []
                n = sum(1 for _ in reader)
            record.update(format='csv', rows=n, columns=headers)
        inventory.append(record)
    hashes = {r['source_file']: r['sha256'] for r in inventory}

    def get(source, keys):
        node = sources[source]
        for key in keys:
            node = node[key]
        return node

    def pointer(keys):
        return '/' + '/'.join(str(k).replace('~', '~0').replace('/', '~1') for k in keys) if keys else ''

    def original_locator(keys):
        result = '$'
        for key in keys:
            if isinstance(key, int):
                result += f'[{key}]'
            elif re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', str(key)):
                result += '.' + str(key)
            else:
                result += '[' + json.dumps(str(key), ensure_ascii=False) + ']'
        return result

    def ascii_key(key, index):
        text = str(key)
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', text):
            return text
        if re.fullmatch(r'[0-9]+', text):
            return 'n_' + text
        return f'k{index:03d}'

    def copy(node, source, original, output):
        rel = (Path('求解结果') / '数据' / source).as_posix()
        if isinstance(node, dict):
            result = {}
            for i, (key, val) in enumerate(node.items()):
                if source.startswith('04c_') and key in ('trajectory','segments','segment_ledger','records','selected_full_ledgers',
                                                         'native_path_rc','sh_m','native_profiles','path_profiles',
                                                         'route_s_h_m','profile_z_m'):
                    continue
                if source == '04b_二维引擎.json' and key in ('ledger','leader','port_hydrograph'):
                    continue
                if source == '04_敏感性.json' and key in ('s_km','z_m','h_m','v_m_s'):
                    continue
                target = ascii_key(key, i)
                if target in result:
                    target += f'_{i}'
                result[target] = copy(val, source, original+[key], output+'.'+target)
            return result
        if isinstance(node, list):
            return [copy(v, source, original+[i], output+f'[{i}]') for i, v in enumerate(node)]
        value = clean_number(node)
        provenance[output] = {'source_file': rel, 'source_locator': original_locator(original),
                              'source_json_pointer': pointer(original), 'source_sha256': hashes[rel],
                              'operation': 'direct copy', 'value': value}
        return value

    values = {'schema_version': 1, 'scope': 'Current computed results; interval targets and limitations remain distinct.',
              'q0': {}, 'q1': {}, 'q2': {}, 'q3': {}, 'q4': {}, 'external': {}, 'csv_inventory': {}}

    def add(section, name, source, *keys):
        values[section][name] = copy(get(source, list(keys)), source, list(keys), f'$.{section}.{name}')

    add('q0','station','00_站点元信息.json')
    for key in ('temperature','hazards','dem','landcover','census','event'):
        add('q0',key,'00_数据清单.json',key)
    add('q0','glacier_change','00_冰川dhdt统计.json')
    ranking='01_特征关联排序.json'
    for name,key in [('top5','前5特征'),('prior_removal','去先验对照'),('ranking_agreement','两方案一致性'),
                     ('robustness','排序稳健性'),('seasonal_shift_control','控制组_日历平移'),
                     ('warm_season_rarity','暖季稀有度'),('leave_year_auc','留一年AUC'),
                     ('classifier_auc','测试AUC'),('candidate_count','候选特征数')]:
        add('q1',name,ranking,key)
    for name,key in [('critic_top5','前5特征'),('critic_weights','权重')]:
        add('q1',name,'01_CRITIC交叉核对.json',key)
    add('q1','boundary_layer','01_边界层结构.json')
    add('q1','hydrostatic_correction','01_静力层压订正.json')
    add('q1','preimpact_anomaly','01b_冲击前窗口异常.json')
    add('q1','unified_association','01c_统一关联前五.json')
    add('q1','weather_modes','01_天气过程模态.json')
    # Raw bootstrap samples are unnecessary for the reader-facing summary.
    energy=sources['01_能量闭合敏感性.json']
    for key in energy:
        if key != '自举C样本':
            add('q1','energy_'+ascii_key(key,list(energy).index(key)),'01_能量闭合敏感性.json',key)
    add('q1','extremes','01_极端事件.json')
    rainfall='02_降雨阈值.json'
    for name,key in [('rain_event_count','独立暴雨事件数'),('rain_upper_a','幂律系数a'),
                     ('rain_upper_b','幂指数b'),('rain_upper_a_ci','参数95pct置信区间_a'),
                     ('rain_upper_b_ci','参数95pct置信区间_b'),('rain_lower','下包络'),
                     ('rain_duration_support_h','频率包络最大支持历时_h')]:
        add('q2',name,rainfall,key)
    for name,source in [('business_screen','02_水源业务判据.json'),('meteorology','02_气象基线检验.json'),
                        ('melt_threshold','02_消融阈值.json'),('historical_domain','02_统计临界域.json'),
                        ('reliability','02_多尺度可靠性.json'),('gully_thresholds','02_沟道临界条件.json'),
                        ('trigger_score','02_触发评估模型.json'),('spatial_contrast','02b_空间对照.json'),
                        ('long_term_temperature','02_长期气温对比.json'),('catchment','02_集水区特征.json'),
                        ('legacy_method_audit','02_旧方法审计.json')]:
        add('q2',name,source)
    add('q2','matched_spatial','02c_共同域空间验证.json')
    for name,source in [('forecast','03_前向预报.json'),
                        ('forecast_member_summary','03_前向预报成员摘要.json'),
                        ('distance','03_空间距离范围.json')]:
        add('q3',name,source)
    for name,source in [('initial_volume','04_初始体积.json'),('centerline_checks','04_中心线校验.json'),
                        ('friction','04_摩擦系数反演.json'),('scenario_summary','04_结果区间.json'),
                        ('intensity','04_强度场.json'),('fan_envelope','04_扇体扩散.json')]:
        add('q4',name,source)
    baseline=sources['04_仿真基线.json']
    for key in ('params','arrival_outlet_s','stop_time_s','termination_reason','runout_km',
                'total_km','reached_outlet','continuity_relative_residual_max','runout_right_censored',
                'v_median_m_s','v_max_m_s','h_median_m','I_max','Cv','rho_m','drop_m','stopped_drop_m'):
        if key in baseline: add('q4','one_d_'+key,'04_仿真基线.json',key)
    add('q4','one_d_unique_configs','04_敏感性.json','unique_runs')
    for name,source in [('finite_inventory_protocol','04c_参数与路径.json'),
                        ('finite_inventory_no_nodes','04c_无节点动力.json'),
                        ('finite_inventory_synthetic','04c_合成库存机制.json'),
                        ('finite_inventory_inverse','04c_逆向瓶颈.json'),
                        ('finite_inventory_checks','04c_数值核验.json'),
                        ('finite_inventory_tests','04c_合成测试.json')]:
        add('q4',name,source)
    routing='04b_二维引擎.json'
    for name,key in [('routing_sections','断面'),('routing_source_mapping','源区落域统计'),
                     ('routing_scenarios','runs'),('routing_coarse','scale90_rerun'),
                     ('routing_time_refinement','time_refinement'),('routing_acceptance','账本验收'),
                     ('routing_unique_count','敏感性情境总数_独立'),('routing_nonarrival','not_reached')]:
        add('q4',name,routing,key)
    for i,item in enumerate(r for r in inventory if r['format']=='csv'):
        name='table_'+str(i)
        values['csv_inventory'][name]={'rows':item['rows'],'columns':item['columns'],'source_file':item['source_file']}
        provenance[f'$.csv_inventory.{name}.rows']={'source_file':item['source_file'],
            'source_sha256':item['sha256'],'operation':'derived','calculation':'Count data records with csv.DictReader, excluding header',
            'value':item['rows']}
    for index, source in enumerate(sorted(sources)):
        if source.startswith(('外部/', '06_')):
            add('external', 'audit_'+str(index), source)
    VALUES_PATH.write_text(json.dumps(values,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    package={'schema_version':1,'summary_file':'求解结果/数据/05_paper_values.json',
             'summary_sha256':sha(VALUES_PATH),'sources':inventory,'entries':provenance,
             'key_policy':'Semantic top-level ASCII aliases; non-ASCII nested source keys become kNNN in original insertion order. Every copied leaf has original locator and JSON Pointer.',
             'refresh':'Rerun after any upstream result changes; no source result is modified.'}
    PROVENANCE_PATH.write_text(json.dumps(package,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'Exported {len(provenance)} evidence-bound leaves from {len(inventory)} source files.')
    print(f'Summary SHA256: {package["summary_sha256"]}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
