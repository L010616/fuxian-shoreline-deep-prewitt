// Fuxian Lake ROUND 4 final blind-test export.
// Model selection, architecture, threshold, seeds, and all comparator checkpoints
// were frozen before this script was released. These six dates must never be
// used for training, tuning, threshold selection, or architecture changes.

var records = [
  {l8_date:'2024-01-13', s2_date:'2024-01-11'},
  {l8_date:'2024-08-24', s2_date:'2024-08-23'},
  {l8_date:'2025-03-04', s2_date:'2025-03-01'},
  {l8_date:'2026-04-24', s2_date:'2026-04-25', l8_product_id:'LC08_L2SP_129043_20260424_20260502_02_T1'},
  {l8_date:'2026-08-14', s2_date:'2026-08-13', l8_product_id:'LC08_L2SP_129043_20260814_20260817_02_T1'},
  {l8_date:'2026-05-26', s2_date:'2026-05-25', l8_product_id:'LC08_L2SP_129043_20260526_20260604_02_T1'}
];

var seed = ee.Geometry.Point([102.8889, 24.5022]);
var searchBox = ee.Geometry.Rectangle([102.78, 24.28, 103.00, 24.72], null, false);
var proj48 = ee.Projection('EPSG:32648');
var l8Transform = [30, 0, 277935, 0, -30, 2727015];
var s2Transform = [10, 0, 277960, 0, -10, 2726990];
var l8Box = ee.Geometry.Rectangle([277935,2694165,294195,2727015], proj48, false);
var s2Box = ee.Geometry.Rectangle([277960,2694180,294190,2726990], proj48, false);

// 2019 JRC extent is used only as a fixed spatial prior and QA zone.
var water2019 = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year', 2019)).first()).select('waterClass').gte(2);
var components = water2019.selfMask().toInt().reduceToVectors({
  geometry:searchBox, scale:30, geometryType:'polygon', eightConnected:true,
  maxPixels:1e7, geometryInNativeProjection:false
});
var lake = ee.Feature(components.filterBounds(seed).first()).geometry().simplify(30);
var roi = lake.buffer(1000,30);
var shore = lake.buffer(300,30).difference(lake.buffer(-300,30),30);

function l8Valid(im) {
  return im.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(im.select('QA_RADSAT').eq(0)).and(im.select('SR_B2').mask())
    .rename('l8_valid').unmask(0);
}
function s2Valid(im) {
  var sc=im.select('SCL');
  return sc.neq(0).and(sc.neq(1)).and(sc.neq(3)).and(sc.lt(8))
    .and(im.select('B2').mask()).rename('s2_valid').unmask(0);
}
function getL8(rec) {
  var start=ee.Date(rec.l8_date);
  var col=ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(roi).filterDate(start,start.advance(1,'day'))
    .filter(ee.Filter.eq('WRS_PATH',129)).filter(ee.Filter.eq('WRS_ROW',43));
  if (rec.l8_product_id) col=col.filter(ee.Filter.eq('LANDSAT_PRODUCT_ID',rec.l8_product_id));
  return ee.Image(col.first());
}
function build(rec) {
  var source=getL8(rec);
  var reflectance=source.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'])
    .multiply(0.0000275).add(-0.2).rename(['B2','B3','B4','B5','B6','B7']);
  var lv=l8Valid(source);
  var s2Start=ee.Date(rec.s2_date);
  var s2col=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi).filterDate(s2Start,s2Start.advance(1,'day'))
    .map(function(im){return im.addBands(s2Valid(im));});
  var s2m=s2col.qualityMosaic('s2_valid');
  var sv=s2col.select('s2_valid').max().unmask(0).rename('s2_valid');
  var joint=lv.and(sv).rename('joint_valid');
  var masks=ee.Image(0).byte().paint(shore,1).rename('shore_zone')
    .addBands(lv).addBands(sv).addBands(joint)
    .addBands(ee.Image(0).byte().paint(roi,1).rename('roi_zone'))
    .toByte().unmask(0).clip(l8Box);
  var l8=reflectance.addBands(lv.rename('valid')).toFloat().unmask(-9999).clip(l8Box);
  var s2=s2m.select(['B2','B3','B4','B8','B11','B12','SCL','s2_valid'])
    .toUint16().unmask(0).clip(s2Box);
  return {l8:l8,s2:s2,masks:masks};
}
function exportRecord(rec) {
  var item=build(rec), l8tag=rec.l8_date.replace(/-/g,''), s2tag=rec.s2_date.replace(/-/g,'');
  Export.image.toDrive({image:item.l8,description:'Fuxian_final_L8_native_'+l8tag,
    folder:'Fuxian_round4_final_blind',region:l8Box,crs:'EPSG:32648',crsTransform:l8Transform,
    maxPixels:1e8,fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true,noData:-9999}});
  Export.image.toDrive({image:item.s2,description:'Fuxian_final_S2_'+s2tag+'_for_'+l8tag,
    folder:'Fuxian_round4_final_blind',region:s2Box,crs:'EPSG:32648',crsTransform:s2Transform,
    maxPixels:1e8,fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true,noData:0}});
  Export.image.toDrive({image:item.masks,description:'Fuxian_final_masks_native_'+l8tag,
    folder:'Fuxian_round4_final_blind',region:l8Box,crs:'EPSG:32648',crsTransform:l8Transform,
    maxPixels:1e8,fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true}});
}

var panel=ui.Panel({style:{width:'420px',padding:'10px'}});
panel.add(ui.Label('Fuxian final blind-test export',{fontSize:'18px',fontWeight:'bold'}));
panel.add(ui.Label('Frozen protocol: six dates, 18 image tasks. Do not tune models with these data.'));
var selector=ui.Select({items:records.map(function(r){return r.l8_date;}),value:records[0].l8_date});
var status=ui.Label('Select a date to inspect.'); panel.add(selector); panel.add(status);
panel.add(ui.Button('Prepare first 3 dates (9 tasks)',function(){
  records.slice(0,3).forEach(exportRecord); status.setValue('First 9 tasks prepared. Start them in Tasks.');
}));
panel.add(ui.Button('Prepare last 3 dates (9 tasks)',function(){
  records.slice(3,6).forEach(exportRecord); status.setValue('Last 9 tasks prepared. Start them in Tasks.');
}));
panel.add(ui.Label('Download all 18 GeoTIFFs only after every task reports COMPLETED.'));
ui.root.insert(0,panel); Map.setCenter(102.8889,24.5022,11);
function show(date) {
  var rec=records.filter(function(r){return r.l8_date===date;})[0]; var item=build(rec);
  Map.layers().reset();
  Map.addLayer(item.l8,{bands:['B4','B3','B2'],min:0,max:0.25},'L8 '+rec.l8_date,true);
  Map.addLayer(item.s2,{bands:['B4','B3','B2'],min:0,max:3000},'S2 '+rec.s2_date,false);
  Map.addLayer(item.masks.select('joint_valid').not().selfMask().clip(shore),{palette:['ff1744']},'Invalid common shore',false);
  Map.addLayer(ee.Image().byte().paint(lake,1,1),{palette:['ffff00']},'2019 prior outline',false);
  status.setValue('Final blind test | L8 '+rec.l8_date+' | S2 '+rec.s2_date);
}
selector.onChange(show); show(records[0].l8_date);
