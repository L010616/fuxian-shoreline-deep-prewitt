// Visual QC and fixed-grid export for two additional independent test pairs.
var records = [
  {l8_date:'2026-01-18', s2_date:'2026-01-20', gap_days:2, joint_shore_valid:0.9974361906243978},
  {l8_date:'2026-02-03', s2_date:'2026-02-04', gap_days:1, joint_shore_valid:0.9932900697495}
];
var seed = ee.Geometry.Point([102.8889, 24.5022]);
var searchBox = ee.Geometry.Rectangle([102.78, 24.28, 103.00, 24.72], null, false);
var proj48 = ee.Projection('EPSG:32648');
var l8Transform = [30, 0, 277935, 0, -30, 2727015];
var s2Transform = [10, 0, 277960, 0, -10, 2726990];
var l8Box = ee.Geometry.Rectangle([277935,2694165,294195,2727015], proj48, false);
var s2Box = ee.Geometry.Rectangle([277960,2694180,294190,2726990], proj48, false);

var water2019 = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year', 2019)).first()).select('waterClass').gte(2);
var components = water2019.selfMask().toInt().reduceToVectors({
  geometry: searchBox, scale: 30, geometryType: 'polygon', eightConnected: true,
  maxPixels: 1e7, geometryInNativeProjection: false
});
var lake = ee.Feature(components.filterBounds(seed).first()).geometry().simplify(30);
var roi = lake.buffer(1000, 30);
var shore = lake.buffer(300, 30).difference(lake.buffer(-300, 30), 30);

function l8Valid(im) {
  return im.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(im.select('QA_RADSAT').eq(0)).and(im.select('SR_B2').mask())
    .rename('l8_valid').unmask(0);
}
function s2Valid(im) {
  var sc = im.select('SCL');
  return sc.neq(0).and(sc.neq(1)).and(sc.neq(3)).and(sc.lt(8))
    .and(im.select('B2').mask()).rename('s2_valid').unmask(0);
}
function build(rec) {
  var ld = ee.Date(rec.l8_date);
  var source = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(seed).filterDate(ld, ld.advance(1,'day'))
    .filter(ee.Filter.eq('WRS_PATH',129)).filter(ee.Filter.eq('WRS_ROW',43)).first();
  var reflectance = source.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'])
    .multiply(0.0000275).add(-0.2).rename(['B2','B3','B4','B5','B6','B7']);
  var lv = l8Valid(source);
  var sd = ee.Date(rec.s2_date);
  var s2col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(searchBox).filterDate(sd, sd.advance(1,'day'))
    .map(function(im){return im.addBands(s2Valid(im));});
  var s2m = s2col.qualityMosaic('s2_valid');
  var sv = s2col.select('s2_valid').max().unmask(0).rename('s2_valid');
  var joint = lv.and(sv).rename('joint_valid');
  var masks = ee.Image(0).byte().paint(shore,1).rename('shore_zone')
    .addBands(lv).addBands(sv).addBands(joint)
    .addBands(ee.Image(0).byte().paint(roi,1).rename('roi_zone'))
    .toByte().unmask(0).clip(l8Box);
  return {
    l8: reflectance.addBands(lv.rename('valid')).toFloat().unmask(-9999).clip(l8Box),
    s2: s2m.select(['B2','B3','B4','B8','B11','B12','SCL','s2_valid'])
      .toUint16().unmask(0).clip(s2Box),
    masks: masks
  };
}
function exportRecord(rec) {
  var item=build(rec), lt=rec.l8_date.replace(/-/g,''), st=rec.s2_date.replace(/-/g,'');
  Export.image.toDrive({image:item.l8,description:'Fuxian_L8_native_'+lt,
    folder:'Fuxian_formal_v1',region:l8Box,crs:'EPSG:32648',crsTransform:l8Transform,
    maxPixels:1e8,fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true,noData:-9999}});
  Export.image.toDrive({image:item.s2,description:'Fuxian_S2_'+st,
    folder:'Fuxian_formal_v1',region:s2Box,crs:'EPSG:32648',crsTransform:s2Transform,
    maxPixels:1e8,fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true,noData:0}});
  Export.image.toDrive({image:item.masks,description:'Fuxian_masks_native_'+lt,
    folder:'Fuxian_formal_v1',region:l8Box,crs:'EPSG:32648',crsTransform:l8Transform,
    maxPixels:1e8,fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true}});
}

var panel=ui.Panel({style:{width:'400px',padding:'10px'}});
panel.add(ui.Label('Additional formal test pairs',{fontSize:'18px',fontWeight:'bold'}));
var selector=ui.Select({items:records.map(function(r){return r.l8_date;}),value:records[0].l8_date});
var status=ui.Label(''); panel.add(selector); panel.add(status);
panel.add(ui.Button('Prepare 6 export tasks',function(){
  records.forEach(exportRecord); status.setValue('Six Drive tasks prepared. Start all in Tasks.');
}));
ui.root.insert(0,panel);
function show(date){
  var rec=records.filter(function(r){return r.l8_date===date;})[0]; var item=build(rec);
  Map.layers().reset();
  Map.addLayer(item.l8,{bands:['B4','B3','B2'],min:0,max:0.25},'L8 '+rec.l8_date,true);
  Map.addLayer(item.s2,{bands:['B4','B3','B2'],min:0,max:3000},'S2 '+rec.s2_date,false);
  Map.addLayer(item.masks.select('joint_valid').not().selfMask().clip(shore),{palette:['ff1744']},'invalid shore',true);
  Map.addLayer(ee.Image().byte().paint(lake,1,1),{palette:['ffff00']},'2019 prior',false);
  status.setValue('L8 '+rec.l8_date+' | S2 '+rec.s2_date+' | gap '+rec.gap_days+
    ' d | joint shore '+(rec.joint_shore_valid*100).toFixed(2)+'%');
  Map.centerObject(lake,11);
}
selector.onChange(show); show(records[0].l8_date);
