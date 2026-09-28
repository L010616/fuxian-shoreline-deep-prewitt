// First-scene repair: preserve the actual Landsat source pixel grid.
// Original 20200102 exports remain diagnostic artifacts, not reference labels.
var seed = ee.Geometry.Point([102.8889, 24.5022]);
var searchBox = ee.Geometry.Rectangle([102.78, 24.28, 103.00, 24.72], null, false);
var water2019 = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year', 2019)).first()).select('waterClass').gte(2);
var components = water2019.selfMask().toInt().reduceToVectors({
  geometry: searchBox, scale: 30, geometryType: 'polygon',
  eightConnected: true, maxPixels: 1e7, geometryInNativeProjection: false
});
var lake = ee.Feature(components.filterBounds(seed).first()).geometry().simplify(30);
var roi = lake.buffer(1000, 30);
var shore = lake.buffer(300, 30).difference(lake.buffer(-300, 30), 30);
var sourceId = 'LANDSAT/LC08/C02/T1_L2/LC08_129043_20200102';
var source = ee.Image(sourceId);
var sourceProjection = source.select('SR_B2').projection();
var exportBox = roi.bounds(30, sourceProjection);
var reflectance = source.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'])
  .multiply(.0000275).add(-.2).rename(['B2','B3','B4','B5','B6','B7']);
var lv = source.select('QA_PIXEL').bitwiseAnd(63).eq(0)
  .and(source.select('QA_RADSAT').eq(0)).and(source.select('SR_B2').mask())
  .rename('l8_valid').unmask(0);
var ids = ['20200102T034141_20200102T034813_T47QRG',
  '20200102T034141_20200102T034813_T47RRH',
  '20200102T034141_20200102T034813_T48QTM',
  '20200102T034141_20200102T034813_T48RTN'];
var s2 = ee.ImageCollection(ids.map(function(id) {
  var im=ee.Image('COPERNICUS/S2_SR_HARMONIZED/'+id), sc=im.select('SCL');
  var v=sc.neq(0).and(sc.neq(1)).and(sc.neq(3)).and(sc.lt(8))
    .and(im.select('B2').mask()).rename('s2_valid').unmask(0);
  return im.addBands(v);
}));
var sv=s2.select('s2_valid').max().unmask(0);
var joint=lv.and(sv).rename('joint_valid');
// Five explicit data bands avoid the accidental four-band RGBA convention.
var masks=ee.Image(0).byte().paint(shore,1).rename('shore_zone')
  .addBands(lv).addBands(sv).addBands(joint)
  .addBands(ee.Image(0).byte().paint(roi,1).rename('roi_zone'))
  .toByte().unmask(0).clip(exportBox);
var l8=reflectance.addBands(lv.rename('valid')).toFloat().unmask(-9999).clip(exportBox);
var panel=ui.Panel({style:{width:'390px',padding:'10px'}});
panel.add(ui.Label('Fuxian pilot: native-grid repair',{fontSize:'18px',fontWeight:'bold'}));
panel.add(ui.Label('2020-01-02 | Re-export L8 and masks only. Keep the existing S2 file.'));
panel.add(ui.Label('No fitted translation is applied. Registration must be checked again after download.'));
var status=ui.Label('Reading source projection...');panel.add(status);
var metadata=ui.Textbox({placeholder:'NATIVE_GRID_METADATA',style:{stretch:'horizontal'}});panel.add(metadata);
var nativeProjection=null;
var button=ui.Button({label:'Prepare 2 native-grid Drive exports',disabled:true,onClick:function(){
  if (!nativeProjection) return;
  Export.image.toDrive({image:l8,description:'Fuxian_L8_native_20200102',
    folder:'Fuxian_pilot',region:exportBox,crs:nativeProjection.crs,
    crsTransform:nativeProjection.transform,maxPixels:1e8,fileFormat:'GeoTIFF',
    formatOptions:{cloudOptimized:true,noData:-9999}});
  Export.image.toDrive({image:masks,description:'Fuxian_masks_native_20200102',
    folder:'Fuxian_pilot',region:exportBox,crs:nativeProjection.crs,
    crsTransform:nativeProjection.transform,maxPixels:1e8,fileFormat:'GeoTIFF',
    formatOptions:{cloudOptimized:true}});
  status.setValue('2 native-grid export tasks prepared. Start each task in Tasks.');
  button.setDisabled(true);
}});panel.add(button);
ui.root.insert(0,panel);
Map.setCenter(102.8889,24.5022,11);
Map.addLayer(reflectance.clip(roi),{bands:['B4','B3','B2'],min:0,max:.2},'L8 source reflectance');
Map.addLayer(s2.qualityMosaic('s2_valid').clip(roi),{bands:['B4','B3','B2'],min:0,max:2000},'S2 reference',false);
sourceProjection.evaluate(function(p,error){
  if(error){status.setValue('Projection error: '+error);return;}
  nativeProjection=p;
  metadata.setValue(JSON.stringify({source:sourceId,projection:p,
    previous_export_grid:[30,0,0,0,-30,0],s2_reexport_required:false,
    masks_bands:['shore_zone','l8_valid','s2_valid','joint_valid','roi_zone'],
    rationale:'Preserve source CRS and affine transform; no arbitrary origin or fitted image shift.'}));
  status.setValue('Native projection ready: '+p.crs+' | '+p.transform.join(', '));
  button.setDisabled(false);
});
