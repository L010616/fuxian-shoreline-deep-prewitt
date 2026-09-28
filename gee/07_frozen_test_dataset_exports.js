// Fuxian Lake exports for the four TEST dates after model freeze.
// Model, loss, threshold, spatial constraint and seeds were frozen on 2026-09-13.
// This script only exports inputs and masks; it does not compute test accuracy.
var records = [{"calendar_gap_days":1,"joint_shore_valid":0.9374735617005566,"l8_date":"2024-02-14","l8_id":"LC08_129043_20240214","l8_shore_valid":0.9982878095396458,"s2_date":"2024-02-15","s2_granule_ids":["20240215T033839_20240215T034800_T47QRG","20240215T033839_20240215T034800_T47RRH","20240215T033839_20240215T034800_T48QTM","20240215T033839_20240215T034800_T48RTN"],"s2_shore_valid":0.9391857563312036,"signed_time_gap_hours":24.272789444444445,"year":2024,"split":"test","qa_group":"core","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false},{"calendar_gap_days":1,"joint_shore_valid":0.9172807242168575,"l8_date":"2024-05-04","l8_id":"LC08_129043_20240504","l8_shore_valid":0.9820598513586963,"s2_date":"2024-05-05","s2_granule_ids":["20240505T033539_20240505T034925_T47QRG","20240505T033539_20240505T034925_T47RRH","20240505T033539_20240505T034925_T48QTM","20240505T033539_20240505T034925_T48RTN"],"s2_shore_valid":0.935918235958984,"signed_time_gap_hours":24.285254444444444,"year":2024,"split":"test","qa_group":"core","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false},{"calendar_gap_days":2,"joint_shore_valid":0.951835063191897,"l8_date":"2025-05-07","l8_id":"LC08_129043_20250507","l8_shore_valid":0.9995264980399335,"s2_date":"2025-05-05","s2_granule_ids":["20250505T033601_20250505T035053_T47QRG","20250505T033601_20250505T035053_T47RRH","20250505T033601_20250505T035053_T48QTM","20250505T033601_20250505T035053_T48RTN"],"s2_shore_valid":0.9522312180280076,"signed_time_gap_hours":-47.710214722222226,"year":2025,"split":"test","qa_group":"core","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false},{"calendar_gap_days":0,"joint_shore_valid":0.9972710869506545,"l8_date":"2025-12-01","l8_id":"LC08_129043_20251201","l8_shore_valid":0.9978345826530814,"s2_date":"2025-12-01","s2_granule_ids":["20251201T034131_20251201T034125_T47QRG","20251201T034131_20251201T034125_T47RRH","20251201T034131_20251201T034125_T48QTM","20251201T034131_20251201T034125_T48RTN"],"s2_shore_valid":0.9994243313323717,"signed_time_gap_hours":0.27161972222222225,"year":2025,"split":"test","qa_group":"core","status":"AUTOMATED_PAIR_QA_PASSED_VISUAL_REVIEW_PENDING","temporal_consistency":"NOT_VERIFIED","registration":"NOT_VERIFIED","label_status":"NOT_CREATED","allow_training":false}];
var wave1 = [0, 1];
var wave2 = [2, 3];
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
  var source = ee.Image('LANDSAT/LC08/C02/T1_L2/' + rec.l8_id);
  var reflectance = source.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'])
    .multiply(0.0000275).add(-0.2).rename(['B2','B3','B4','B5','B6','B7']);
  var lv = l8Valid(source);
  var s2col = ee.ImageCollection(rec.s2_granule_ids.map(function(id) {
    var image = ee.Image('COPERNICUS/S2_SR_HARMONIZED/' + id);
    return image.addBands(s2Valid(image));
  }));
  var s2m = s2col.qualityMosaic('s2_valid');
  var sv = s2col.select('s2_valid').max().unmask(0).rename('s2_valid');
  var joint = lv.and(sv).rename('joint_valid');
  var masks = ee.Image(0).byte().paint(shore, 1).rename('shore_zone')
    .addBands(lv).addBands(sv).addBands(joint)
    .addBands(ee.Image(0).byte().paint(roi, 1).rename('roi_zone'))
    .toByte().unmask(0).clip(l8Box);
  var l8 = reflectance.addBands(lv.rename('valid')).toFloat().unmask(-9999).clip(l8Box);
  var s2 = s2m.select(['B2','B3','B4','B8','B11','B12','SCL','s2_valid'])
    .toUint16().unmask(0).clip(s2Box);
  return {record: rec, l8: l8, s2: s2, masks: masks};
}

function exportRecord(rec) {
  var item = build(rec);
  var l8tag = rec.l8_date.replace(/-/g, '');
  var s2tag = rec.s2_date.replace(/-/g, '');
  Export.image.toDrive({image:item.l8, description:'Fuxian_L8_native_'+l8tag,
    folder:'Fuxian_frozen_test_v1', region:l8Box, crs:'EPSG:32648', crsTransform:l8Transform,
    maxPixels:1e8, fileFormat:'GeoTIFF', formatOptions:{cloudOptimized:true,noData:-9999}});
  Export.image.toDrive({image:item.s2, description:'Fuxian_S2_'+s2tag,
    folder:'Fuxian_frozen_test_v1', region:s2Box, crs:'EPSG:32648', crsTransform:s2Transform,
    maxPixels:1e8, fileFormat:'GeoTIFF', formatOptions:{cloudOptimized:true,noData:0}});
  Export.image.toDrive({image:item.masks, description:'Fuxian_masks_native_'+l8tag,
    folder:'Fuxian_frozen_test_v1', region:l8Box, crs:'EPSG:32648', crsTransform:l8Transform,
    maxPixels:1e8, fileFormat:'GeoTIFF', formatOptions:{cloudOptimized:true}});
}

var panel = ui.Panel({style:{width:'410px',padding:'10px'}});
panel.add(ui.Label('Fuxian frozen TEST exports', {fontSize:'18px',fontWeight:'bold'}));
panel.add(ui.Label('Four test dates opened only after the model-freeze decision.'));
var selector = ui.Select({items:records.map(function(r){return r.l8_date;}), value:records[0].l8_date});
panel.add(selector);
var status = ui.Label('Select a date to inspect.'); panel.add(status);
panel.add(ui.Button('Prepare TEST wave 1: 6 export tasks', function() {
  wave1.forEach(function(i) { exportRecord(records[i]); });
  status.setValue('TEST wave 1 prepared: 2 dates, 6 Drive tasks. Start them in Tasks.');
}));
panel.add(ui.Button('Prepare TEST wave 2: 6 export tasks', function() {
  wave2.forEach(function(i) { exportRecord(records[i]); });
  status.setValue('TEST wave 2 prepared: 2 dates, 6 Drive tasks. Start them after wave 1.');
}));
panel.add(ui.Label('Exports use the verified fixed grids. Do not inspect model accuracy until final labels are frozen.'));
ui.root.insert(0, panel);
Map.setCenter(102.8889, 24.5022, 11);

function show(date) {
  var rec = records.filter(function(r) { return r.l8_date === date; })[0];
  var item = build(rec);
  Map.layers().reset();
  Map.addLayer(item.l8, {bands:['B4','B3','B2'],min:0,max:0.25}, 'L8 '+rec.l8_date, true);
  Map.addLayer(item.s2, {bands:['B4','B3','B2'],min:0,max:3000}, 'S2 '+rec.s2_date, false);
  Map.addLayer(item.masks.select('joint_valid').not().selfMask().clip(shore),
    {palette:['ff1744']}, 'invalid common shore', false);
  Map.addLayer(ee.Image().byte().paint(lake,1,1), {palette:['ffff00']}, '2019 prior outline', false);
  status.setValue(rec.split+' | L8 '+rec.l8_date+' | S2 '+rec.s2_date+
    ' | common shore '+(rec.joint_shore_valid*100).toFixed(2)+'%');
}
selector.onChange(show);
show(records[0].l8_date);
