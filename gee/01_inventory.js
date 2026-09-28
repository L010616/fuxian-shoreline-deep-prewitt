// Fuxian Lake pilot: inventory and pre-2020 screening geometry.
// The 2019 JRC outline is a screening prior, NEVER a reference label.
// No full-period maximum extent is used. No export is started automatically.
var cfg = {start: '2020-01-01', end: '2026-01-01', year: 2020};
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
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(seed).filterDate(cfg.start, cfg.end);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi).filterDate(cfg.start, cfg.end);
function l8valid(im) {
  return im.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(im.select('QA_RADSAT').eq(0))
    .and(im.select('SR_B2').mask()).rename('valid').unmask(0);
}
function metadata(im) {
  im = ee.Image(im);
  return ee.Feature(null, {
    id: im.id(), date: im.date().format('YYYY-MM-dd'),
    year: im.date().get('year'), month: im.date().get('month'),
    scene_cloud: im.get('CLOUD_COVER'),
    path: im.get('WRS_PATH'), row: im.get('WRS_ROW')
  });
}
var yearly = ee.FeatureCollection(ee.List.sequence(2020, 2025).map(function(y) {
  var a = ee.Date.fromYMD(y, 1, 1), b = a.advance(1, 'year');
  return ee.Feature(null, {year:y, landsat8_scenes:l8.filterDate(a,b).size(),
    landsat8_cloud_le80:l8.filterDate(a,b).filter(ee.Filter.lte('CLOUD_COVER',80)).size(),
    sentinel2_granules:s2.filterDate(a,b).size(),
    sentinel2_dates:s2.filterDate(a,b).map(function(im){
      return im.set('day', im.date().format('YYYY-MM-dd'));
    }).aggregate_count_distinct('day')});
}));
var panel = ui.Panel({style:{width:'370px', padding:'8px'}});
panel.add(ui.Label('Fuxian Lake pilot | inventory', {fontWeight:'bold',fontSize:'18px'}));
panel.add(ui.Label('2019 JRC outline = screening prior only, not ground truth.'));
var status = ui.Label('Computing inventory...'); panel.add(status);
var yearSelect = ui.Select({items:['2020','2021','2022','2023','2024','2025'], value:'2020'});
panel.add(ui.Label('Quality screening year')); panel.add(yearSelect);
var output = ui.Textbox({placeholder:'Machine-readable result will appear here',style:{stretch:'horizontal'}});
panel.add(output);
panel.add(ui.Button('Screen selected year', function(){
  status.setValue('Computing 30 m shore-zone valid fraction...');
  var a=ee.Date.fromYMD(Number(yearSelect.getValue()),1,1), b=a.advance(1,'year');
  var subset=l8.filterDate(a,b).filter(ee.Filter.lte('CLOUD_COVER',80));
  var records=ee.FeatureCollection(subset.toList(subset.size()).map(function(obj){
    var im=ee.Image(obj);
    var valid=l8valid(im).reduceRegion({reducer:ee.Reducer.mean(),geometry:shore,
      scale:30,maxPixels:1e7,tileScale:2}).get('valid');
    var nearby=s2.filterDate(im.date().advance(-3,'day'), im.date().advance(4,'day'));
    return metadata(im).set({shore_valid:valid,s2_nearby_granules:nearby.size()});
  })).sort('date');
  records.evaluate(function(result,error){
    if(error){status.setValue('ERROR: '+error);return;}
    var rows=result.features.map(function(f){return f.properties;});
    output.setValue(JSON.stringify(rows));
    print('QUALITY_'+yearSelect.getValue(),rows);
    status.setValue('Done: '+rows.length+' scenes. Fractions are automatic QA, require visual review.');
  });
}));
ui.root.insert(0,panel);
Map.centerObject(seed,10);
Map.addLayer(ee.Image().byte().paint(shore,1),{palette:['ffd54f']},'Screening shore zone +/-300m',false);
Map.addLayer(ee.Image().byte().paint(lake,1,2),{palette:['00ffff']},'JRC 2019 provisional outline');
Map.addLayer(ee.Image().byte().paint(roi,1,1),{palette:['ff4081']},'Common ROI +1000m',false);
ee.Dictionary({outline_area_km2:lake.area(30).divide(1e6),
  roi_area_km2:roi.area(30).divide(1e6),
  inventory:yearly.toList(6).map(function(f){return ee.Feature(f).toDictionary();})
}).evaluate(function(result,error){
  if(error){status.setValue('ERROR: '+error);return;}
  output.setValue(JSON.stringify(result)); print('INVENTORY', result);
  status.setValue('Inventory ready. Inspect geometry and screen years.');
});
Export.table.toDrive({collection:yearly,description:'Fuxian_inventory_2020_2025',
  folder:'Fuxian_pilot',fileFormat:'CSV'});

// Batch all six years once, with a separate result field.
var batchStatus = ui.Label('Computing all-year shore QA...'); panel.add(batchStatus);
var batchOutput = ui.Textbox({placeholder:'ALL_YEAR_QA_JSON',style:{stretch:'horizontal'}});
panel.add(batchOutput);
var allSubset = l8.filter(ee.Filter.lte('CLOUD_COVER',80));
var allQuality = ee.FeatureCollection(allSubset.toList(allSubset.size()).map(function(obj){
  var im=ee.Image(obj);
  var day=ee.Date.parse('YYYY-MM-dd',im.date().format('YYYY-MM-dd'));
  var valid=l8valid(im).reduceRegion({reducer:ee.Reducer.mean(),geometry:shore,
    scale:30,maxPixels:1e7,tileScale:2}).get('valid');
  var nearby=s2.filterDate(day.advance(-3,'day'),day.advance(4,'day'));
  return metadata(im).set({shore_valid:valid,s2_nearby_granules:nearby.size()});
})).sort('date');
allQuality.evaluate(function(result,error){
  if(error){batchStatus.setValue('ERROR: '+error);return;}
  var rows=result.features.map(function(f){return f.properties;});
  batchOutput.setValue(JSON.stringify(rows));
  print('ALL_YEAR_SHORE_QA',rows);
  batchStatus.setValue('All-year QA complete: '+rows.length+' scenes.');
});
