// Prediction-blind replacement for the unusable 2025-09-28 confirmation date.
// Select the highest-quality completely unused L8/S2 pair from 2020-2025.
// Model outputs must remain unopened until the replacement reference
// has been screened and the six-date set has been frozen.

var lakeSeed = ee.Geometry.Point([102.8889, 24.5022]);
var searchBox = ee.Geometry.Rectangle([102.78, 24.28, 103.00, 24.72], null, false);
var proj48 = ee.Projection('EPSG:32648');
var l8Transform = [30, 0, 277935, 0, -30, 2727015];
var s2Transform = [10, 0, 277960, 0, -10, 2726990];
var l8Box = ee.Geometry.Rectangle([277935, 2694165, 294195, 2727015], proj48, false);
var s2Box = ee.Geometry.Rectangle([277960, 2694180, 294190, 2726990], proj48, false);

var water2019 = ee.Image(
  ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
    .filter(ee.Filter.eq('year', 2019)).first()
).select('waterClass').gte(2);
var components = water2019.selfMask().toInt().reduceToVectors({
  geometry: searchBox, scale: 30, geometryType: 'polygon', eightConnected: true,
  maxPixels: 1e7, geometryInNativeProjection: false
});
var lake = ee.Feature(components.filterBounds(lakeSeed).first()).geometry().simplify(30);
var roi = lake.buffer(1000, 30);
var shore = lake.buffer(300, 30).difference(lake.buffer(-300, 30), 30);

var excludedDates = ee.List([
  // Training
  '2020-01-02','2020-01-18','2020-05-09','2020-07-28',
  '2021-01-04','2021-02-05','2021-05-12','2021-10-19',
  '2022-03-12','2022-04-13','2022-08-19','2022-12-09',
  // Validation
  '2023-01-26','2023-03-31','2023-05-18',
  // Primary blind test
  '2024-01-13','2024-08-24','2025-03-04',
  '2026-04-24','2026-05-26','2026-08-14',
  // First post-2019 confirmation and rejected opened date
  '2021-03-25','2021-12-22','2022-02-08','2023-11-10',
  '2024-04-18','2026-02-19','2024-01-29',
  // Auxiliary/application dates
  '2024-02-14','2024-04-02','2024-05-04','2024-10-27',
  '2025-01-31','2025-04-21','2025-05-07','2025-10-14',
  '2025-12-01','2026-01-18','2026-02-03',
  // Confirmation set 2, including the rejected date
  '2021-04-10','2021-09-01','2022-04-29','2023-06-03',
  '2025-02-16','2025-09-28',
  // Earlier 2018 confirmation
  '2018-01-28','2018-02-13','2018-03-01','2018-04-02','2018-05-20','2018-11-28'
]);

function l8Valid(image) {
  return image.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(image.select('QA_RADSAT').eq(0))
    .and(image.select('SR_B2').mask()).rename('l8_valid').unmask(0);
}

function s2Valid(image) {
  var scl = image.select('SCL');
  return scl.neq(0).and(scl.neq(1)).and(scl.neq(3)).and(scl.lt(8))
    .and(image.select('B2').mask()).rename('s2_valid').unmask(0);
}

function clearFraction(mask, scale) {
  var value = mask.rename('clear').reduceRegion({
    reducer: ee.Reducer.mean(), geometry: shore, scale: scale,
    maxPixels: 1e8, tileScale: 4
  }).get('clear');
  return ee.Number(ee.Algorithms.If(value, value, 0));
}

function bestS2Pair(l8Date) {
  var choices = ee.FeatureCollection(ee.List.sequence(-5, 5).map(function(offset) {
    offset = ee.Number(offset);
    var day = l8Date.advance(offset, 'day');
    var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(roi).filterDate(day, day.advance(1, 'day'))
      .map(function(image) { return image.addBands(s2Valid(image)); });
    var count = collection.size();
    var valid = ee.Image(ee.Algorithms.If(
      count.gt(0), collection.select('s2_valid').max().unmask(0),
      ee.Image.constant(0).rename('s2_valid')
    ));
    var clear = clearFraction(valid, 20);
    return ee.Feature(null, {
      s2_date: day.format('YYYY-MM-dd'), s2_clear_shore: clear,
      day_gap: offset.abs(), s2_scene_count: count,
      pair_score: clear.subtract(offset.abs().multiply(0.025))
    });
  })).filter(ee.Filter.gt('s2_scene_count', 0)).sort('pair_score', false);
  var fallback = ee.Feature(null, {
    s2_date: '1970-01-01', s2_clear_shore: 0, day_gap: 99,
    s2_scene_count: 0, pair_score: -999
  });
  return ee.Feature(ee.Algorithms.If(choices.size().gt(0), choices.first(), fallback));
}

var l8Collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(roi)
  .filterDate('2020-01-01', '2026-01-01')
  .filter(ee.Filter.eq('WRS_PATH', 129))
  .filter(ee.Filter.eq('WRS_ROW', 43))
  .filter(ee.Filter.lt('CLOUD_COVER', 80))
  // First perform a cheap metadata-only exclusion and retain only the 8
  // clearest unused Landsat scenes. Sentinel-2 matching and reduceRegion are
  // then evaluated for this small list, preventing GEE memory exhaustion.
  .map(function(image) {
    var dateText = image.date().format('YYYY-MM-dd');
    return image.set({
      date_text: dateText,
      excluded_num: ee.Number(ee.Algorithms.If(
        excludedDates.contains(dateText), 1, 0
      ))
    });
  })
  .filter(ee.Filter.eq('excluded_num', 0))
  .sort('CLOUD_COVER')
  .limit(8);

var candidates = ee.FeatureCollection(l8Collection.map(function(image) {
  var date = image.date();
  var dateText = date.format('YYYY-MM-dd');
  var pair = bestS2Pair(date);
  var l8Clear = clearFraction(l8Valid(image), 30);
  var s2Clear = ee.Number(pair.get('s2_clear_shore'));
  var gap = ee.Number(pair.get('day_gap'));
  var score = l8Clear.multiply(0.55).add(s2Clear.multiply(0.45))
    .subtract(gap.multiply(0.02));
  return ee.Feature(null, {
    l8_date: dateText,
    l8_system_id: image.get('system:id'),
    l8_product_id: image.get('LANDSAT_PRODUCT_ID'),
    l8_scene_cloud_pct: image.get('CLOUD_COVER'),
    l8_clear_shore: l8Clear,
    s2_date: pair.get('s2_date'),
    s2_clear_shore: s2Clear,
    s2_scene_count: pair.get('s2_scene_count'),
    day_gap: gap,
    quality_score: score,
    excluded_num: 0
  });
}))
  .filter(ee.Filter.eq('excluded_num', 0))
  .filter(ee.Filter.gt('s2_scene_count', 0))
  .filter(ee.Filter.gte('l8_clear_shore', 0.60))
  .filter(ee.Filter.gte('s2_clear_shore', 0.60))
  .sort('quality_score', false);

var fallback = ee.Feature(null, {
  l8_date: 'MISSING', s2_date: 'MISSING', quality_score: -999
});
var selected = ee.Feature(ee.Algorithms.If(candidates.size().gt(0), candidates.first(), fallback));
print('SET2_REPLACEMENT_SELECTED', selected);

function build(record) {
  var l8Source = ee.Image(record.l8_system_id);
  var reflectance = l8Source
    .select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'])
    .multiply(0.0000275).add(-0.2)
    .rename(['B2','B3','B4','B5','B6','B7']);
  var l8Mask = l8Valid(l8Source);
  var s2Start = ee.Date(record.s2_date);
  var s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi).filterDate(s2Start, s2Start.advance(1, 'day'))
    .map(function(image) { return image.addBands(s2Valid(image)); });
  var s2Mosaic = s2Collection.qualityMosaic('s2_valid');
  var s2Mask = s2Collection.select('s2_valid').max().unmask(0).rename('s2_valid');
  var masks = ee.Image.constant(0).byte().paint(shore, 1).rename('shore_zone')
    .addBands(l8Mask).addBands(s2Mask)
    .addBands(l8Mask.and(s2Mask).rename('joint_valid'))
    .addBands(ee.Image.constant(0).byte().paint(roi, 1).rename('roi_zone'))
    .toByte().unmask(0).clip(l8Box);
  return {
    l8: reflectance.addBands(l8Mask.rename('valid')).toFloat().unmask(-9999).clip(l8Box),
    s2: s2Mosaic.select(['B2','B3','B4','B8','B11','B12','SCL','s2_valid'])
      .toUint16().unmask(0).clip(s2Box),
    masks: masks
  };
}

var panel = ui.Panel({style: {width: '455px', padding: '10px'}});
panel.add(ui.Label('Set-2 quality replacement', {fontSize: '18px', fontWeight: 'bold'}));
var status = ui.Label('Selecting an unused pair with at least 60% clear shoreline...');
panel.add(status);
ui.root.insert(0, panel);
Map.setCenter(102.8889, 24.5022, 11);

selected.evaluate(function(feature, error) {
  if (error) {
    status.setValue('Selection failed: ' + error);
    return;
  }
  var record = feature.properties;
  if (!record || record.l8_date === 'MISSING') {
    status.setValue('STOP: no unused pair passed the 60% quality threshold.');
    return;
  }
  status.setValue(
    'Replacement locked: L8 ' + record.l8_date + ' / S2 ' + record.s2_date +
    ' | clear ' + (100 * record.l8_clear_shore).toFixed(1) + '% / ' +
    (100 * record.s2_clear_shore).toFixed(1) + '%'
  );
  var products = build(record);
  var l8Tag = record.l8_date.replace(/-/g, '');
  var s2Tag = record.s2_date.replace(/-/g, '');
  var folder = 'Fuxian_edge_confirmation_set2_replacement';

  panel.add(ui.Button('Prepare replacement: 3 GeoTIFF tasks', function() {
    Export.image.toDrive({
      image: products.l8,
      description: 'Fuxian_edgeconfirm_set2_L8_native_' + l8Tag,
      folder: folder, region: l8Box, crs: 'EPSG:32648',
      crsTransform: l8Transform, maxPixels: 1e8, fileFormat: 'GeoTIFF',
      formatOptions: {cloudOptimized: true, noData: -9999}
    });
    Export.image.toDrive({
      image: products.s2,
      description: 'Fuxian_edgeconfirm_set2_S2_' + s2Tag + '_for_' + l8Tag,
      folder: folder, region: s2Box, crs: 'EPSG:32648',
      crsTransform: s2Transform, maxPixels: 1e8, fileFormat: 'GeoTIFF',
      formatOptions: {cloudOptimized: true, noData: 0}
    });
    Export.image.toDrive({
      image: products.masks,
      description: 'Fuxian_edgeconfirm_set2_masks_native_' + l8Tag,
      folder: folder, region: l8Box, crs: 'EPSG:32648',
      crsTransform: l8Transform, maxPixels: 1e8, fileFormat: 'GeoTIFF',
      formatOptions: {cloudOptimized: true}
    });
    status.setValue('Three replacement tasks prepared. Start them in Tasks.');
  }));

  panel.add(ui.Button('Prepare replacement manifest CSV', function() {
    Export.table.toDrive({
      collection: ee.FeatureCollection([selected]),
      description: 'Fuxian_edgeconfirm_set2_replacement_manifest',
      folder: folder,
      fileFormat: 'CSV'
    });
    status.setValue('Replacement manifest task prepared.');
  }));

  Map.layers().reset();
  Map.addLayer(products.l8, {bands: ['B4','B3','B2'], min: 0, max: 0.25},
    'L8 ' + record.l8_date, true);
  Map.addLayer(products.s2, {bands: ['B4','B3','B2'], min: 0, max: 3000},
    'S2 ' + record.s2_date, false);
  Map.addLayer(products.masks.select('joint_valid').not().selfMask().clip(shore),
    {palette: ['ff1744']}, 'Invalid common shore', false);
});
