// Fuxian Lake: post-2019, date-disjoint confirmatory exports for the edge branch.
//
// LOCKED PROTOCOL
// 1. Candidate dates are selected only from Landsat/Sentinel image quality and
//    temporal proximity. No model prediction or accuracy is read.
// 2. Every date already used by training, validation, blind testing, earlier
//    confirmation, auxiliary testing, or the 2020-2025 mapping examples is excluded.
// 3. One date is selected from each of six non-overlapping time windows.
// 4. Keep all six selected dates. Screen low-confidence shoreline segments before
//    viewing predictions. Never use this set for training, model selection, threshold
//    selection, early stopping, or reference editing based on model output.
// 5. The script prepares 18 GeoTIFF tasks: 6 dates x (Landsat 8 + Sentinel-2 + masks).

var seed = ee.Geometry.Point([102.8889, 24.5022]);
var searchBox = ee.Geometry.Rectangle([102.78, 24.28, 103.00, 24.72], null, false);
var proj48 = ee.Projection('EPSG:32648');
var l8Transform = [30, 0, 277935, 0, -30, 2727015];
var s2Transform = [10, 0, 277960, 0, -10, 2726990];
var l8Box = ee.Geometry.Rectangle([277935,2694165,294195,2727015], proj48, false);
var s2Box = ee.Geometry.Rectangle([277960,2694180,294190,2726990], proj48, false);

// Fixed prior used by every model configuration. All confirmation dates are later
// than the prior year, so no future spatial information is used.
var water2019 = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year', 2019)).first()).select('waterClass').gte(2);
var components = water2019.selfMask().toInt().reduceToVectors({
  geometry: searchBox, scale: 30, geometryType: 'polygon', eightConnected: true,
  maxPixels: 1e7, geometryInNativeProjection: false
});
var lake = ee.Feature(components.filterBounds(seed).first()).geometry().simplify(30);
var roi = lake.buffer(1000, 30);
var shore = lake.buffer(300, 30).difference(lake.buffer(-300, 30), 30);

// Dates already opened or used anywhere in the current project.
var excludedDates = ee.List([
  // 12 training dates
  '2020-01-02','2020-01-18','2020-05-09','2020-07-28',
  '2021-01-04','2021-02-05','2021-05-12','2021-10-19',
  '2022-03-12','2022-04-13','2022-08-19','2022-12-09',
  // 3 validation dates
  '2023-01-26','2023-03-31','2023-05-18',
  // 6 primary blind-test dates
  '2024-01-13','2024-08-24','2025-03-04',
  '2026-04-24','2026-05-26','2026-08-14',
  // Earlier post-2019 auxiliary/unseen tests
  '2024-02-14','2024-04-02','2024-05-04','2024-10-27',
  '2025-01-31','2025-04-21','2025-05-07','2025-10-14','2025-12-01',
  '2026-01-18','2026-02-03',
  // Earlier 2018 edge-confirmation set (listed for protocol completeness)
  '2018-01-28','2018-02-13','2018-03-01','2018-04-02','2018-05-20','2018-11-28'
]);

// Six independent temporal windows. One quality-ranked pair is selected per window.
var windows = [
  // The first window is deliberately wider because late-2020 scenes around the
  // lake are frequently cloudy. Previously used dates remain excluded below.
  {name:'W1_2020_2021early', start:'2020-01-01', end:'2021-06-01'},
  {name:'W2_2021_late', start:'2021-06-01', end:'2022-01-01'},
  {name:'W3_2022_early',start:'2022-01-01', end:'2022-08-01'},
  {name:'W4_2023_late', start:'2023-06-01', end:'2024-01-01'},
  {name:'W5_2024',      start:'2024-01-01', end:'2025-01-01'},
  {name:'W6_2025_2026', start:'2025-01-01', end:'2026-09-01'}
];

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
function clearFraction(mask, geometry, scale) {
  return ee.Number(mask.rename('clear').reduceRegion({
    reducer: ee.Reducer.mean(), geometry: geometry, scale: scale,
    maxPixels: 1e8, tileScale: 4
  }).get('clear'));
}
function bestS2Pair(l8Date) {
  var offsets = ee.List.sequence(-2, 2);
  var choices = ee.FeatureCollection(offsets.map(function(offset) {
    offset = ee.Number(offset);
    var day = l8Date.advance(offset, 'day');
    var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(roi).filterDate(day, day.advance(1, 'day'))
      .map(function(im) { return im.addBands(s2Valid(im)); });
    var exists = col.size().gt(0);
    var valid = ee.Image(ee.Algorithms.If(
      exists, col.select('s2_valid').max().unmask(0), ee.Image(0).rename('s2_valid')
    ));
    var frac = clearFraction(valid, shore, 20);
    var score = frac.subtract(offset.abs().multiply(0.025));
    return ee.Feature(null, {
      s2_date: day.format('YYYY-MM-dd'), s2_clear_shore: frac,
      day_gap: offset.abs(), s2_scene_count: col.size(), pair_score: score
    });
  })).filter(ee.Filter.gt('s2_scene_count', 0)).sort('pair_score', false);
  return ee.Feature(choices.first());
}
function bestInWindow(win) {
  var l8col = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(roi).filterDate(win.start, win.end)
    .filter(ee.Filter.eq('WRS_PATH',129)).filter(ee.Filter.eq('WRS_ROW',43))
    .filter(ee.Filter.lt('CLOUD_COVER',70));
  var candidates = ee.FeatureCollection(l8col.map(function(im) {
    var d = im.date();
    var dateText = d.format('YYYY-MM-dd');
    var pair = bestS2Pair(d);
    var l8frac = clearFraction(l8Valid(im), shore, 30);
    var s2frac = ee.Number(pair.get('s2_clear_shore'));
    var gap = ee.Number(pair.get('day_gap'));
    var score = l8frac.multiply(0.55).add(s2frac.multiply(0.45))
      .subtract(gap.multiply(0.02));
    return ee.Feature(null, {
      window: win.name,
      l8_date: dateText,
      l8_system_id: im.get('system:id'),
      l8_product_id: im.get('LANDSAT_PRODUCT_ID'),
      l8_scene_cloud_pct: im.get('CLOUD_COVER'),
      l8_clear_shore: l8frac,
      s2_date: pair.get('s2_date'),
      s2_clear_shore: s2frac,
      day_gap: gap,
      quality_score: score,
      excluded: excludedDates.contains(dateText)
    });
  })).filter(ee.Filter.eq('excluded', false))
    // Always retain the highest-quality unused pair in each locked window.
    // Clear fractions are reported for quality screening but do not delete a
    // whole date after the confirmation protocol has been fixed.
    .sort('quality_score', false);
  return ee.Feature(candidates.first());
}

var selected = ee.FeatureCollection(windows.map(bestInWindow));
print('LOCKED_POST2019_CONFIRMATORY_PAIRS', selected);
print('PAIR_COUNT_MUST_EQUAL_6', selected.size());

function getL8(rec) {
  return ee.Image(rec.l8_system_id);
}
function build(rec) {
  var source = getL8(rec);
  var reflectance = source.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'])
    .multiply(0.0000275).add(-0.2).rename(['B2','B3','B4','B5','B6','B7']);
  var lv = l8Valid(source);
  var s2Start = ee.Date(rec.s2_date);
  var s2col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi).filterDate(s2Start,s2Start.advance(1,'day'))
    .map(function(im){ return im.addBands(s2Valid(im)); });
  var s2m = s2col.qualityMosaic('s2_valid');
  var sv = s2col.select('s2_valid').max().unmask(0).rename('s2_valid');
  var joint = lv.and(sv).rename('joint_valid');
  var masks = ee.Image(0).byte().paint(shore,1).rename('shore_zone')
    .addBands(lv).addBands(sv).addBands(joint)
    .addBands(ee.Image(0).byte().paint(roi,1).rename('roi_zone'))
    .toByte().unmask(0).clip(l8Box);
  var l8 = reflectance.addBands(lv.rename('valid')).toFloat().unmask(-9999).clip(l8Box);
  var s2 = s2m.select(['B2','B3','B4','B8','B11','B12','SCL','s2_valid'])
    .toUint16().unmask(0).clip(s2Box);
  return {l8:l8,s2:s2,masks:masks,source:source};
}
function exportRecord(rec) {
  var item = build(rec);
  var l8tag = rec.l8_date.replace(/-/g,'');
  var s2tag = rec.s2_date.replace(/-/g,'');
  Export.image.toDrive({image:item.l8,
    description:'Fuxian_edgeconfirm_post2019_L8_native_'+l8tag,
    folder:'Fuxian_edge_confirmatory_post2019',region:l8Box,
    crs:'EPSG:32648',crsTransform:l8Transform,maxPixels:1e8,
    fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true,noData:-9999}});
  Export.image.toDrive({image:item.s2,
    description:'Fuxian_edgeconfirm_post2019_S2_'+s2tag+'_for_'+l8tag,
    folder:'Fuxian_edge_confirmatory_post2019',region:s2Box,
    crs:'EPSG:32648',crsTransform:s2Transform,maxPixels:1e8,
    fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true,noData:0}});
  Export.image.toDrive({image:item.masks,
    description:'Fuxian_edgeconfirm_post2019_masks_native_'+l8tag,
    folder:'Fuxian_edge_confirmatory_post2019',region:l8Box,
    crs:'EPSG:32648',crsTransform:l8Transform,maxPixels:1e8,
    fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true}});
}

var panel = ui.Panel({style:{width:'455px',padding:'10px'}});
panel.add(ui.Label('Post-2019 edge confirmation exports',
  {fontSize:'18px',fontWeight:'bold'}));
panel.add(ui.Label('Quality-only selection; all previously used dates are excluded.'));
panel.add(ui.Label('Wait until the six selected pairs appear below.'));
var status = ui.Label('Selecting six date-disjoint pairs on the server...');
panel.add(status);
ui.root.insert(0,panel);
Map.setCenter(102.8889,24.5022,11);

selected.evaluate(function(fc, err) {
  if (err) {
    status.setValue('Selection failed: '+err);
    return;
  }
  var records = fc.features.map(function(f) { return f.properties; });
  if (records.length !== 6) {
    status.setValue('STOP: expected 6 pairs, obtained '+records.length+'. Do not export.');
    return;
  }
  records.sort(function(a,b) { return a.window.localeCompare(b.window); });
  status.setValue('Locked six pairs: '+records.map(function(r){return r.l8_date;}).join(', '));
  panel.add(ui.Label(records.map(function(r){
    return r.window+': L8 '+r.l8_date+' / S2 '+r.s2_date+
      ' | clear '+(100*r.l8_clear_shore).toFixed(1)+'% / '+
      (100*r.s2_clear_shore).toFixed(1)+'%';
  }).join('\n'),{whiteSpace:'pre'}));
  var selector = ui.Select({
    items:records.map(function(r){return r.l8_date;}), value:records[0].l8_date
  });
  panel.add(selector);
  panel.add(ui.Button('Prepare first 3 dates: 9 GeoTIFF tasks',function(){
    records.slice(0,3).forEach(exportRecord);
    status.setValue('First 9 tasks prepared. Start them in Tasks.');
  }));
  panel.add(ui.Button('Prepare last 3 dates: 9 GeoTIFF tasks',function(){
    records.slice(3,6).forEach(exportRecord);
    status.setValue('Last 9 tasks prepared. Start them in Tasks.');
  }));
  panel.add(ui.Button('Optional: prepare selection manifest CSV',function(){
    Export.table.toDrive({collection:selected,
      description:'Fuxian_edgeconfirm_post2019_selection_manifest',
      folder:'Fuxian_edge_confirmatory_post2019',fileFormat:'CSV'});
    status.setValue('One optional manifest task prepared.');
  }));
  panel.add(ui.Label('Required download: all 18 GeoTIFF files. The CSV is recommended.'));

  function show(date) {
    var rec = records.filter(function(r){return r.l8_date===date;})[0];
    var item = build(rec);
    Map.layers().reset();
    Map.addLayer(item.l8,{bands:['B4','B3','B2'],min:0,max:0.25},
      'L8 '+rec.l8_date,true);
    Map.addLayer(item.s2,{bands:['B4','B3','B2'],min:0,max:3000},
      'S2 '+rec.s2_date,false);
    Map.addLayer(item.masks.select('joint_valid').not().selfMask().clip(shore),
      {palette:['ff1744']},'Invalid common shore',false);
    Map.addLayer(ee.Image().byte().paint(lake,1,1),
      {palette:['ffff00']},'2019 prior outline',false);
    print('SELECTED_PAIR',rec);
  }
  selector.onChange(show);
  show(records[0].l8_date);
});
