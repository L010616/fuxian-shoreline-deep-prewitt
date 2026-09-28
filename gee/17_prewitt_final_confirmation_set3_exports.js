// Fuxian Lake: final independent confirmation set for the selected deep Prewitt branch.
//
// LOCKED PROTOCOL
// 1. Select six unused Landsat 8 / Sentinel-2 pairs from a fixed shortlist
//    established from the pre-model image-quality catalogue.
// 2. Every Landsat and Sentinel-2 date previously used or opened is excluded.
// 3. Do not view model predictions before reference screening is frozen.
// 4. Do not use this set for training, architecture selection, threshold tuning,
//    early stopping, or checkpoint selection.
// 5. Required exports: 18 GeoTIFFs = 6 dates x (Landsat 8 + Sentinel-2 + masks).

// A tiny seed box is used instead of Geometry.Point because some Code Editor
// sessions currently fail while decorating the Point constructor.
var lakeSeed = ee.Geometry.Rectangle(
  [102.8888, 24.5021, 102.8890, 24.5023], null, false
);
var searchBox = ee.Geometry.Rectangle([102.78, 24.28, 103.00, 24.72], null, false);
var proj48 = ee.Projection('EPSG:32648');
var l8Transform = [30, 0, 277935, 0, -30, 2727015];
var s2Transform = [10, 0, 277960, 0, -10, 2726990];
var l8Box = ee.Geometry.Rectangle([277935, 2694165, 294195, 2727015], proj48, false);
var s2Box = ee.Geometry.Rectangle([277960, 2694180, 294190, 2726990], proj48, false);

// Fixed 2019 spatial prior, identical to the formal experiments.
var water2019 = ee.Image(
  ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
    .filter(ee.Filter.eq('year', 2019))
    .first()
).select('waterClass').gte(2);

var components = water2019.selfMask().toInt().reduceToVectors({
  geometry: searchBox,
  scale: 30,
  geometryType: 'polygon',
  eightConnected: true,
  maxPixels: 1e7,
  geometryInNativeProjection: false
});

var lake = ee.Feature(components.filterBounds(lakeSeed).first())
  .geometry().simplify(30);
var roi = lake.buffer(1000, 30);
var shore = lake.buffer(300, 30)
  .difference(lake.buffer(-300, 30), 30);

// Complete project-level exclusion list. Dates remain excluded even if an
// earlier pair was rejected after quality screening, because it has been opened.
var excludedL8Dates = ee.List([
  // Training dates (12)
  '2020-01-02', '2020-01-18', '2020-05-09', '2020-07-28',
  '2021-01-04', '2021-02-05', '2021-05-12', '2021-10-19',
  '2022-03-12', '2022-04-13', '2022-08-19', '2022-12-09',

  // Validation dates (3)
  '2023-01-26', '2023-03-31', '2023-05-18',

  // Primary blind-test dates (6)
  '2024-01-13', '2024-08-24', '2025-03-04',
  '2026-04-24', '2026-05-26', '2026-08-14',

  // First post-2019 edge-confirmation set (6) and its rejected opened date
  '2021-03-25', '2021-12-22', '2022-02-08', '2023-11-10',
  '2024-04-18', '2026-02-19', '2024-01-29',

  // Earlier auxiliary / unseen tests and application-only dates
  '2024-02-14', '2024-04-02', '2024-05-04', '2024-10-27',
  '2025-01-31', '2025-04-21', '2025-05-07', '2025-10-14',
  '2025-12-01', '2026-01-18', '2026-02-03',

  // Second edge-confirmation set, including its rejected cloudy date
  '2021-04-10', '2021-09-01', '2022-04-29', '2023-02-11',
  '2023-06-03', '2025-02-16', '2025-09-28',

  // Earlier 2018 edge-confirmation dates
  '2018-01-28', '2018-02-13', '2018-03-01',
  '2018-04-02', '2018-05-20', '2018-11-28'
]);

// Sentinel-2 dates are excluded separately so a new Landsat date cannot reuse
// a reference image that has already been opened in an earlier experiment.
var excludedS2Dates = ee.List([
  '2020-01-02','2020-01-17','2020-05-11','2020-07-28',
  '2021-01-04','2021-02-05','2021-03-27','2021-04-11','2021-05-11',
  '2021-09-03','2021-10-19','2021-12-22',
  '2022-02-10','2022-03-12','2022-04-13','2022-04-26','2022-08-19','2022-12-09',
  '2023-01-26','2023-02-10','2023-04-01','2023-05-18','2023-05-31','2023-11-12',
  '2024-01-11','2024-01-31','2024-02-15','2024-04-20','2024-05-05',
  '2024-08-23','2024-10-27',
  '2025-02-19','2025-03-01','2025-05-05','2025-09-27','2025-10-14','2025-12-01',
  '2026-01-20','2026-02-04','2026-02-19','2026-04-25','2026-05-25','2026-08-13',
  '2018-01-27','2018-02-11','2018-03-03','2018-04-02','2018-05-22','2018-11-28'
]);

// Fixed prediction-blind shortlist from the existing image-quality catalogue.
// None of these Landsat dates has been used by any model experiment.
var candidateDates = [
  '2020-02-03', '2020-02-19', '2020-03-06',
  '2021-11-20', '2022-01-07', '2022-11-07',
  '2022-11-23', '2023-01-10', '2024-12-30'
];

function l8Valid(image) {
  return image.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(image.select('QA_RADSAT').eq(0))
    .and(image.select('SR_B2').mask())
    .rename('l8_valid').unmask(0);
}

function s2Valid(image) {
  var scl = image.select('SCL');
  return scl.neq(0).and(scl.neq(1)).and(scl.neq(3)).and(scl.lt(8))
    .and(image.select('B2').mask())
    .rename('s2_valid').unmask(0);
}

function clearFraction(mask, geometry, scale) {
  var value = mask.rename('clear').reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: geometry,
    scale: scale,
    maxPixels: 1e8,
    tileScale: 4
  }).get('clear');
  return ee.Number(ee.Algorithms.If(value, value, 0));
}

// Search up to five days before and after each Landsat acquisition. A safe
// fallback feature prevents the previous "Image has no bands" failure when no
// Sentinel-2 scene exists for a candidate date.
function bestS2Pair(l8Date) {
  var offsets = ee.List.sequence(-5, 5);
  var choices = ee.FeatureCollection(offsets.map(function(offset) {
    offset = ee.Number(offset);
    var day = l8Date.advance(offset, 'day');
    var dayText = day.format('YYYY-MM-dd');
    var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(roi)
      .filterDate(day, day.advance(1, 'day'))
      .map(function(image) {
        return image.addBands(s2Valid(image));
      });
    var count = collection.size();
    var valid = ee.Image(ee.Algorithms.If(
      count.gt(0),
      collection.select('s2_valid').max().unmask(0),
      ee.Image.constant(0).rename('s2_valid')
    ));
    var clear = clearFraction(valid, shore, 20);
    var gap = offset.abs();
    return ee.Feature(null, {
      s2_date: dayText,
      s2_clear_shore: clear,
      day_gap: gap,
      s2_scene_count: count,
      pair_score: clear.subtract(gap.multiply(0.025)),
      has_pair_num: ee.Number(count),
      excluded_s2_num: ee.Number(ee.Algorithms.If(
        excludedS2Dates.contains(dayText), 1, 0
      ))
    });
  }))
    .filter(ee.Filter.eq('excluded_s2_num', 0))
    .filter(ee.Filter.gt('s2_scene_count', 0))
    .sort('pair_score', false);

  var fallback = ee.Feature(null, {
    s2_date: '1970-01-01',
    s2_clear_shore: 0,
    day_gap: 99,
    s2_scene_count: 0,
    pair_score: -999,
    has_pair_num: 0
  });
  return ee.Feature(ee.Algorithms.If(choices.size().gt(0), choices.first(), fallback));
}

var candidates = ee.FeatureCollection(candidateDates.map(function(dateText) {
  var date = ee.Date(dateText);
  var l8Collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(roi)
    .filterDate(date, date.advance(1, 'day'))
    .filter(ee.Filter.eq('WRS_PATH', 129))
    .filter(ee.Filter.eq('WRS_ROW', 43));
  var image = ee.Image(l8Collection.first());
  var pair = bestS2Pair(date);
  var l8Clear = clearFraction(l8Valid(image), shore, 30);
  var s2Clear = ee.Number(pair.get('s2_clear_shore'));
  var gap = ee.Number(pair.get('day_gap'));
  var jointScore = l8Clear.multiply(0.55)
    .add(s2Clear.multiply(0.45))
    .subtract(gap.multiply(0.02));
  return ee.Feature(null, {
    window: 'fixed_prediction_blind_shortlist',
    season: 'mixed',
    l8_date: dateText,
    l8_system_id: image.get('system:id'),
    l8_product_id: image.get('LANDSAT_PRODUCT_ID'),
    l8_scene_cloud_pct: image.get('CLOUD_COVER'),
    l8_clear_shore: l8Clear,
    s2_date: pair.get('s2_date'),
    s2_clear_shore: s2Clear,
    day_gap: gap,
    quality_score: jointScore,
    has_s2_pair_num: pair.get('has_pair_num'),
    valid_candidate_num: 1
  });
}))
  .filter(ee.Filter.gt('has_s2_pair_num', 0))
  .filter(ee.Filter.gte('l8_clear_shore', 0.50))
  .filter(ee.Filter.gte('s2_clear_shore', 0.15))
  .sort('quality_score', false);

var selected = candidates.limit(6);
print('PREWITT_FINAL_CONFIRMATION_SET3_PAIRS', selected);
print('PAIR_COUNT_MUST_EQUAL_6', selected.size());

function build(record) {
  var l8Source = ee.Image(record.l8_system_id);
  var reflectance = l8Source
    .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
    .multiply(0.0000275).add(-0.2)
    .rename(['B2', 'B3', 'B4', 'B5', 'B6', 'B7']);
  var l8Mask = l8Valid(l8Source);

  var s2Start = ee.Date(record.s2_date);
  var s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate(s2Start, s2Start.advance(1, 'day'))
    .map(function(image) {
      return image.addBands(s2Valid(image));
    });
  var s2Mosaic = s2Collection.qualityMosaic('s2_valid');
  var s2Mask = s2Collection.select('s2_valid').max().unmask(0).rename('s2_valid');
  var jointMask = l8Mask.and(s2Mask).rename('joint_valid');

  var masks = ee.Image.constant(0).byte().paint(shore, 1).rename('shore_zone')
    .addBands(l8Mask)
    .addBands(s2Mask)
    .addBands(jointMask)
    .addBands(ee.Image.constant(0).byte().paint(roi, 1).rename('roi_zone'))
    .toByte().unmask(0).clip(l8Box);

  var l8Export = reflectance.addBands(l8Mask.rename('valid'))
    .toFloat().unmask(-9999).clip(l8Box);
  var s2Export = s2Mosaic
    .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'SCL', 's2_valid'])
    .toUint16().unmask(0).clip(s2Box);

  return {l8: l8Export, s2: s2Export, masks: masks};
}

function exportRecord(record) {
  var products = build(record);
  var l8Tag = record.l8_date.replace(/-/g, '');
  var s2Tag = record.s2_date.replace(/-/g, '');
  var folder = 'Fuxian_prewitt_confirmation_set3';

  Export.image.toDrive({
    image: products.l8,
    description: 'Fuxian_prewitt_confirm_set3_L8_native_' + l8Tag,
    folder: folder,
    region: l8Box,
    crs: 'EPSG:32648',
    crsTransform: l8Transform,
    maxPixels: 1e8,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: -9999}
  });

  Export.image.toDrive({
    image: products.s2,
    description: 'Fuxian_prewitt_confirm_set3_S2_' + s2Tag + '_for_' + l8Tag,
    folder: folder,
    region: s2Box,
    crs: 'EPSG:32648',
    crsTransform: s2Transform,
    maxPixels: 1e8,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: 0}
  });

  Export.image.toDrive({
    image: products.masks,
    description: 'Fuxian_prewitt_confirm_set3_masks_native_' + l8Tag,
    folder: folder,
    region: l8Box,
    crs: 'EPSG:32648',
    crsTransform: l8Transform,
    maxPixels: 1e8,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true}
  });
}

var panel = ui.Panel({style: {width: '470px', padding: '10px'}});
panel.add(ui.Label('Fuxian Prewitt final confirmation set 3', {
  fontSize: '18px', fontWeight: 'bold'
}));
panel.add(ui.Label(
  'Six completely unused pairs selected from a fixed prediction-blind shortlist. ' +
  'Selection is based only on image quality and temporal proximity; ' +
  'all earlier Landsat and Sentinel-2 dates are excluded.'
));
var status = ui.Label('Selecting six independent pairs on the server...');
panel.add(status);
ui.root.insert(0, panel);
Map.setCenter(102.8889, 24.5022, 11);

selected.evaluate(function(featureCollection, error) {
  if (error) {
    status.setValue('Selection failed: ' + JSON.stringify(error));
    return;
  }

  var records = featureCollection.features.map(function(feature) {
    return feature.properties;
  });
  records.sort(function(a, b) {
    return Number(b.quality_score) - Number(a.quality_score);
  });

  var invalid = records.filter(function(record) {
    return Number(record.valid_candidate_num || 0) <= 0 ||
      record.l8_date === 'MISSING';
  });
  var uniqueDates = {};
  var uniqueS2Dates = {};
  records.forEach(function(record) { uniqueDates[record.l8_date] = true; });
  records.forEach(function(record) { uniqueS2Dates[record.s2_date] = true; });

  if (records.length !== 6 || invalid.length > 0 ||
      Object.keys(uniqueDates).length !== 6 || Object.keys(uniqueS2Dates).length !== 6) {
    status.setValue(
      'STOP: six valid unique pairs were not found. Do not export. ' +
      'Valid records: ' + (records.length - invalid.length) + '/6.'
    );
    return;
  }

  status.setValue('Six unused pairs locked: ' + records.map(function(record) {
    return record.l8_date;
  }).join(', '));

  panel.add(ui.Label(records.map(function(record) {
    return record.window + ' [' + record.season + ']  L8 ' + record.l8_date +
      ' / S2 ' + record.s2_date + ' | clear ' +
      (100 * record.l8_clear_shore).toFixed(1) + '% / ' +
      (100 * record.s2_clear_shore).toFixed(1) + '% | gap ' +
      record.day_gap + ' d';
  }).join('\n'), {whiteSpace: 'pre'}));

  var selector = ui.Select({
    items: records.map(function(record) { return record.l8_date; }),
    value: records[0].l8_date
  });
  panel.add(selector);

  panel.add(ui.Button('Prepare first 3 dates: 9 GeoTIFF tasks', function() {
    records.slice(0, 3).forEach(exportRecord);
    status.setValue('First 9 tasks prepared. Start them in the Tasks tab.');
  }));

  panel.add(ui.Button('Prepare last 3 dates: 9 GeoTIFF tasks', function() {
    records.slice(3, 6).forEach(exportRecord);
    status.setValue('Last 9 tasks prepared. Start them in the Tasks tab.');
  }));

  panel.add(ui.Button('Prepare selection manifest CSV', function() {
    Export.table.toDrive({
      collection: selected,
      description: 'Fuxian_prewitt_confirm_set3_selection_manifest',
      folder: 'Fuxian_prewitt_confirmation_set3',
      fileFormat: 'CSV'
    });
    status.setValue('One manifest CSV task prepared.');
  }));

  panel.add(ui.Label(
    'Required: download all 18 GeoTIFF files. The manifest CSV is strongly recommended.'
  ));

  function show(dateText) {
    var record = records.filter(function(item) {
      return item.l8_date === dateText;
    })[0];
    var products = build(record);
    Map.layers().reset();
    Map.addLayer(products.l8, {
      bands: ['B4', 'B3', 'B2'], min: 0, max: 0.25
    }, 'L8 ' + record.l8_date, true);
    Map.addLayer(products.s2, {
      bands: ['B4', 'B3', 'B2'], min: 0, max: 3000
    }, 'S2 ' + record.s2_date, false);
    Map.addLayer(
      products.masks.select('joint_valid').not().selfMask().clip(shore),
      {palette: ['ff1744']}, 'Invalid common shore', false
    );
    Map.addLayer(
      ee.Image.constant(0).byte().paint(lake, 1, 1),
      {palette: ['ffff00']}, '2019 prior outline', false
    );
    print('SET3_SELECTED_PAIR', record);
  }

  selector.onChange(show);
  show(records[0].l8_date);
});

