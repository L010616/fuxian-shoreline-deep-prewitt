// Fuxian Lake: second fully independent confirmation set for the edge branch.
//
// LOCKED PROTOCOL
// 1. Select one unused Landsat 8 / Sentinel-2 pair from each of six fixed
//    seasonal windows. Selection uses image quality and temporal proximity only.
// 2. Dates used by training, validation, blind testing, the first confirmation
//    set, auxiliary tests, or mapping examples are explicitly excluded.
// 3. Do not view model predictions before reference screening is frozen.
// 4. Do not use this set for training, architecture selection, threshold tuning,
//    early stopping, or checkpoint selection.
// 5. Required exports: 18 GeoTIFFs = 6 dates x (Landsat 8 + Sentinel-2 + masks).

var lakeSeed = ee.Geometry.Point([102.8889, 24.5022]);
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
var excludedDates = ee.List([
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

  // Earlier 2018 edge-confirmation dates
  '2018-01-28', '2018-02-13', '2018-03-01',
  '2018-04-02', '2018-05-20', '2018-11-28'
]);

// Three dry-season and three wet-season windows. All windows end in 2025 so
// the new set does not enlarge the manuscript's temporal scope into 2026.
var windows = [
  {name: 'D1_2020_2021_dry', start: '2020-10-01', end: '2021-05-01', season: 'dry'},
  {name: 'W1_2021_wet',      start: '2021-06-01', end: '2021-10-01', season: 'wet'},
  {name: 'D2_2022_dry',      start: '2022-01-01', end: '2022-06-01', season: 'dry'},
  {name: 'W2_2023_wet',      start: '2023-06-01', end: '2023-11-01', season: 'wet'},
  {name: 'D3_2024_2025_dry', start: '2024-10-01', end: '2025-05-01', season: 'dry'},
  {name: 'W3_2025_wet',      start: '2025-05-01', end: '2025-10-01', season: 'wet'}
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

// Search up to three days before and after each Landsat acquisition. A safe
// fallback feature prevents the previous "Image has no bands" failure when no
// Sentinel-2 scene exists for a candidate date.
function bestS2Pair(l8Date) {
  var offsets = ee.List.sequence(-3, 3);
  var choices = ee.FeatureCollection(offsets.map(function(offset) {
    offset = ee.Number(offset);
    var day = l8Date.advance(offset, 'day');
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
      s2_date: day.format('YYYY-MM-dd'),
      s2_clear_shore: clear,
      day_gap: gap,
      s2_scene_count: count,
      pair_score: clear.subtract(gap.multiply(0.025)),
      has_pair_num: ee.Number(count)
    });
  })).filter(ee.Filter.gt('s2_scene_count', 0)).sort('pair_score', false);

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

function bestInWindow(windowSpec) {
  var l8Collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(roi)
    .filterDate(windowSpec.start, windowSpec.end)
    .filter(ee.Filter.eq('WRS_PATH', 129))
    .filter(ee.Filter.eq('WRS_ROW', 43))
    .filter(ee.Filter.lt('CLOUD_COVER', 80));

  var candidates = ee.FeatureCollection(l8Collection.map(function(image) {
    var date = image.date();
    var dateText = date.format('YYYY-MM-dd');
    var pair = bestS2Pair(date);
    var l8Clear = clearFraction(l8Valid(image), shore, 30);
    var s2Clear = ee.Number(pair.get('s2_clear_shore'));
    var gap = ee.Number(pair.get('day_gap'));
    var jointScore = l8Clear.multiply(0.55)
      .add(s2Clear.multiply(0.45))
      .subtract(gap.multiply(0.02));

    return ee.Feature(null, {
      window: windowSpec.name,
      season: windowSpec.season,
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
      excluded_num: ee.Number(ee.Algorithms.If(
        excludedDates.contains(dateText), 1, 0
      ))
    });
  }))
    .filter(ee.Filter.eq('excluded_num', 0))
    .filter(ee.Filter.gt('has_s2_pair_num', 0))
    .sort('quality_score', false);

  var fallback = ee.Feature(null, {
    window: windowSpec.name,
    season: windowSpec.season,
    l8_date: 'MISSING',
    s2_date: 'MISSING',
    quality_score: -999,
    valid_candidate_num: 0
  });
  var selected = ee.Feature(ee.Algorithms.If(
    candidates.size().gt(0), candidates.first(), fallback
  ));
  return selected.set('valid_candidate_num', candidates.size());
}

var selected = ee.FeatureCollection(windows.map(bestInWindow));
print('EDGE_CONFIRMATION_SET2_PAIRS', selected);
print('PAIR_COUNT_MUST_EQUAL_6', selected.size());
print('ALL_PAIRS_MUST_BE_VALID', selected.aggregate_min('valid_candidate_num'));

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
  var folder = 'Fuxian_edge_confirmation_set2';

  Export.image.toDrive({
    image: products.l8,
    description: 'Fuxian_edgeconfirm_set2_L8_native_' + l8Tag,
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
    description: 'Fuxian_edgeconfirm_set2_S2_' + s2Tag + '_for_' + l8Tag,
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
    description: 'Fuxian_edgeconfirm_set2_masks_native_' + l8Tag,
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
panel.add(ui.Label('Fuxian edge confirmation set 2', {
  fontSize: '18px', fontWeight: 'bold'
}));
panel.add(ui.Label(
  'Six unused dates: three dry-season and three wet-season pairs. ' +
  'Selection is based only on image quality and temporal proximity.'
));
var status = ui.Label('Selecting six independent pairs on the server...');
panel.add(status);
ui.root.insert(0, panel);
Map.setCenter(102.8889, 24.5022, 11);

selected.evaluate(function(featureCollection, error) {
  if (error) {
    status.setValue('Selection failed: ' + error);
    return;
  }

  var records = featureCollection.features.map(function(feature) {
    return feature.properties;
  });
  records.sort(function(a, b) {
    return a.window.localeCompare(b.window);
  });

  var invalid = records.filter(function(record) {
    return Number(record.valid_candidate_num || 0) <= 0 ||
      record.l8_date === 'MISSING';
  });
  var uniqueDates = {};
  records.forEach(function(record) { uniqueDates[record.l8_date] = true; });

  if (records.length !== 6 || invalid.length > 0 || Object.keys(uniqueDates).length !== 6) {
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
      description: 'Fuxian_edgeconfirm_set2_selection_manifest',
      folder: 'Fuxian_edge_confirmation_set2',
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
    print('SET2_SELECTED_PAIR', record);
  }

  selector.onChange(show);
  show(records[0].l8_date);
});
