"""Physical visual carriers, installed identically for every paired strategy.

Textures are scene materials, never pasted onto a camera observation. Saved old
observations remain immutable. Online policies only see a public action menu.
"""
import hashlib
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from .scene import _model,_surface


VISUAL_POLICIES = ('uao_spsa','iccv_uada_spsa','iccv_upa_spsa')
KINDS = dict(uao_spsa='uao_sphere',iccv_uada_spsa='iccv_uada_patch',iccv_upa_spsa='iccv_upa_patch')


def load_bundle(path):
    path=Path(path).resolve()
    bundle=json.loads(path.read_text())
    if bundle.get('status')!='frozen_after_training':raise ValueError('Baseline textures must be trained and frozen before evaluation')
    if (bundle.get('implementation_hash_at_start') is not None and
        bundle.get('implementation_hash_at_start')!=bundle.get('implementation_hash_at_end')):
        raise ValueError('Development texture optimization changed code; use a stable frozen training run')
    files={}
    for policy in VISUAL_POLICIES:
        item=bundle['textures'][policy];source=path.parent/item['path']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Frozen baseline texture changed: '+policy)
        files[policy]=source
    return bundle,files,hashlib.sha256(path.read_bytes()).hexdigest()


def add_visual_assets(scene,textures,bundle_hash=None):
    world_file=Path(scene['world']);root=world_file.parent
    tree=ET.parse(str(world_file));world=tree.getroot().find('world')
    assets=root/'assets';material_file=assets/'materials/scripts/pilot.material'
    material=material_file.read_text()
    for policy in VISUAL_POLICIES:
        image=assets/'materials/textures'/('baseline_'+policy+'.png')
        shutil.copyfile(str(textures[policy]),str(image))
        name='baseline_'+policy
        material+='\nmaterial Pilot/%s\n{\n technique\n {\n  pass\n  {\n   ambient 1 1 1\n   diffuse 1 1 1\n   emissive 0.15 0.15 0.15\n   cull_hardware none\n   texture_unit\n   {\n    texture %s.png\n   }\n  }\n }\n}\n'%(name,name)
        for i,device in enumerate(scene['truth']['devices'],1):
            model_name='visual_%s_%02d'%(policy,i)
            # All assets begin behind the service wall, invisible to both
            # cameras and front-side LiDAR; all policies receive identical assets.
            link=_model(world,model_name,[0,5.,.3,0,0,0])
            if policy=='uao_spsa':
                for kind in ('visual','collision'):
                    item=ET.SubElement(link,kind,name='sphere')
                    shape=ET.SubElement(ET.SubElement(item,'geometry'),'sphere')
                    ET.SubElement(shape,'radius').text='.03'
                    if kind=='visual':
                        script=ET.SubElement(ET.SubElement(item,'material'),'script')
                        ET.SubElement(script,'uri').text=(assets/'materials/scripts').as_uri()
                        ET.SubElement(script,'uri').text=(assets/'materials/textures').as_uri()
                        ET.SubElement(script,'name').text='Pilot/'+name
                dimensions=dict(radius_m=.03,texture_pixels=[128,128])
            else:
                _surface(link,'patch',assets,name,[0,0,0],.30,.30)
                item=ET.SubElement(link,'collision',name='patch_board')
                box=ET.SubElement(ET.SubElement(item,'geometry'),'box')
                ET.SubElement(box,'size').text='.30 .008 .30'
                dimensions=dict(width_m=.30,height_m=.30,texture_pixels=[128,128])
            action='deploy_%s_%02d'%(policy,i)
            scene['catalog']['editable_actions'].append(dict(action_id=action,carrier=model_name,kind=KINDS[policy],
                description='Deploy the fixed trained visual carrier beside station %02d; appearance changes, no identity or physical fact change'%i,
                dimensions=dimensions))
            scene['action_states'][action]=dict(model_name=model_name,
                position=[device['position'][0]+.78,1.82,1.35],yaw=0.)
    material_file.write_text(material)
    tree.write(str(world_file),encoding='utf-8',xml_declaration=True)
    scene['catalog']['visual_carriers']=dict(bundle_sha256=bundle_hash,
        scene_materials_only=True,shared_initial_assets=True,
        methods_are_api_adaptations=True,
        limitation='ICCV logits/gradients and 7-DoF arm outputs are unavailable; SPSA optimizes high-level API decision surrogates. UAO uses one-step surrogate, not original complete-rollout four-loss training.')
    return scene
