// Fuxian Lake formal-dataset exports for TRAIN and VALIDATION dates only.
// Test dates are intentionally absent and remain sealed until the method is fixed.
// The 2019 JRC geometry is a spatial prior, never a target-date reference label.
var records = [{"calendar_gap_days":1,"joint_shore_valid":0.9999832137781309,"l8_date":"2020-01-18","l8_id":"LC08_129043_20200118","l8_shore_valid":0.999983213819016,"s2_date":"2020-01-17","s2_granule_ids":["20200117T034059_20200117T034126_T47QRG","20200117T034059_20200117T034126_T47RRH","20200117T034059_20200117T034126_T48QTM","20200117T034059_20200117T034126_T48RTN"],"s2_shore_valid":1,"signed_time_gap_hours":-23.730629166666667,"year":2020,"split":"train","qa_group":"core","status":"FROZEN_FOR_EXPORT_AND_REFERENCE_REVIEW","allow_training":false},{"calendar_gap_days":2,"joint_shore_valid":0.8123242466113229,"l8_date":"2020-07-28","l8_id":"LC08_129043_20200728","l8_shore_valid":0.8851265023055989,"s2_date":"2020-07-30","s2_granule_ids":["20200730T033541_20200730T034218_T47QRG","20200730T033541_20200730T034218_T47RRH","20200730T033541_20200730T034218_T48QTM","20200730T033541_20200730T034218_T48RTN"],"s2_shore_valid":0.9240397129095265,"signed_time_gap_hours":48.2764775,"year":2020,"split":"train","qa_group":"partial","status":"FROZEN_FOR_EXPORT_AND_REFERENCE_REVIEW","allow_training":false},{"calendar_gap_days":2,"joint_shore_valid":0.9938028560275375,"l8_date":"2021-01-04","l8_id":"LC08_129043_20210104","l8_shore_valid":0.9937568572372171,"s2_date":"2021-01-06","s2_granule_ids":["20210106T034131_20210106T034741_T47QRG","20210106T034131_20210106T034741_T47RRH","20210106T034131_20210106T034741_T48QTM","20210106T034131_20210106T034741_T48RTN"],"s2_shore_valid":0.9999496413343927,"signed_time_gap_hours":48.26957222222222,"year":2021,"split":"train","qa_group":"core","status":"FROZEN_FOR_EXPORT_AND_REFERENCE_REVIEW","allow_training":false},{"calendar_gap_days":1,"joint_shore_valid":0.7827992571932253,"l8_date":"2021-10-19","l8_id":"LC08_129043_20211019","l8_shore_valid":0.8634209170428915,"s2_date":"2021-10-18","s2_granule_ids":["20211018T033739_20211018T035016_T47QRG","20211018T033739_20211018T035016_T47RRH","20211018T033739_20211018T035016_T48QTM","20211018T033739_20211018T035016_T48RTN"],"s2_shore_valid":0.9167236191357667,"signed_time_gap_hours":-23.733004444444443,"year":2021,"split":"train","qa_group":"partial","status":"FROZEN_FOR_EXPORT_AND_REFERENCE_REVIEW","allow_training":false},{"calendar_gap_days":3,"joint_shore_valid":0.9449716049537128,"l8_date":"2022-04-13","l8_id":"LC08_129043_20220413","l8_shore_valid":0.9998153520091774,"s2_date":"2022-04-16","s2_granule_ids":["20220416T033529_20220416T034805_T47QRG","20220416T033529_20220416T034805_T47RRH","20220416T033529_20220416T034805_T48QTM","20220416T033529_20220416T034805_T48RTN"],"s2_shore_valid":0.9451394671724037,"signed_time_gap_hours":72.2754575,"year":2022,"split":"train","qa_group":"core","status":"FROZEN_FOR_EXPORT_AND_REFERENCE_REVIEW","allow_training":false},{"calendar_gap_days":2,"joint_shore_valid":0.9976370924157217,"l8_date":"2022-12-09","l8_id":"LC08_129043_20221209","l8_shore_valid":0.999983213819016,"s2_date":"2022-12-07","s2_granule_ids":["20221207T034131_20221207T035101_T47QRG","20221207T034131_20221207T035101_T47RRH","20221207T034131_20221207T035101_T48QTM","20221207T034131_20221207T035101_T48RTN"],"s2_shore_valid":0.9976538786375908,"signed_time_gap_hours":-47.73509638888889,"year":2022,"split":"train","qa_group":"core","status":"FROZEN_FOR_EXPORT_AND_REFERENCE_REVIEW","allow_training":false},{"calendar_gap_days":3,"joint_shore_valid":0.8968261533286426,"l8_date":"2023-05-18","l8_id":"LC08_129043_20230518","l8_shore_valid":0.8959092208598652,"s2_date":"2023-05-21","s2_granule_ids":["20230521T033539_20230521T034748_T47QRG","20230521T033539_20230521T034748_T47RRH","20230521T033539_20230521T034748_T48QTM","20230521T033539_20230521T034748_T48RTN"],"s2_shore_valid":1,"signed_time_gap_hours":72.28752972222222,"year":2023,"split":"validation","qa_group":"partial","status":"FROZEN_FOR_EXPORT_AND_REFERENCE_REVIEW","allow_training":false}];
var wave1 = [0,1,2,3];
var wave2 = [4,5,6];
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
    folder:'Fuxian_round4_trainval', region:l8Box, crs:'EPSG:32648', crsTransform:l8Transform,
    maxPixels:1e8, fileFormat:'GeoTIFF', formatOptions:{cloudOptimized:true,noData:-9999}});
  Export.image.toDrive({image:item.s2, description:'Fuxian_S2_'+s2tag,
    folder:'Fuxian_round4_trainval', region:s2Box, crs:'EPSG:32648', crsTransform:s2Transform,
    maxPixels:1e8, fileFormat:'GeoTIFF', formatOptions:{cloudOptimized:true,noData:0}});
  Export.image.toDrive({image:item.masks, description:'Fuxian_masks_native_'+l8tag,
    folder:'Fuxian_round4_trainval', region:l8Box, crs:'EPSG:32648', crsTransform:l8Transform,
    maxPixels:1e8, fileFormat:'GeoTIFF', formatOptions:{cloudOptimized:true}});
}

var panel = ui.Panel({style:{width:'410px',padding:'10px'}});
panel.add(ui.Label('Fuxian formal dataset exports', {fontSize:'18px',fontWeight:'bold'}));
panel.add(ui.Label('Only seven added train/validation dates. Final blind test remains sealed.'));
var selector = ui.Select({items:records.map(function(r){return r.l8_date;}), value:records[0].l8_date});
panel.add(selector);
var status = ui.Label('Select a date to inspect.'); panel.add(status);
panel.add(ui.Button('Prepare wave 1: 12 export tasks', function() {
  wave1.forEach(function(i) { exportRecord(records[i]); });
  status.setValue('Wave 1 prepared: 4 dates, 12 Drive tasks. Start them in Tasks.');
}));
panel.add(ui.Button('Prepare wave 2: 9 export tasks', function() {
  wave2.forEach(function(i) { exportRecord(records[i]); });
  status.setValue('Wave 2 prepared: 3 dates, 9 Drive tasks. Start them after wave 1.');
}));
panel.add(ui.Label('Exports use the fixed grids. No final-test image task is created.'));
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
