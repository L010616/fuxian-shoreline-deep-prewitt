// Q2 extension, STAGE 1: prediction-blind external-lake image inventory.
// Paste the whole file into the Google Earth Engine Code Editor and click Run.
// Scan ONE lake at a time. After the quality scan finishes, use the buttons in
// the lower-left panel to create a CSV task or three candidate GeoTIFF tasks.
// Candidate images are for image-quality/reference review, NOT a locked test.
// Do not run a trained model on these lakes before final dates are frozen.
//
// Data: Landsat 8 Collection 2 Level-2, Sentinel-2 SR Harmonized, and the
// 2019 JRC YearlyHistory waterClass 2/3. Date range: 2020-01-01 to 2025-12-31.
// Each scanned year contributes at most three low-cloud Landsat scenes per
// season, before local shoreline quality and <=4-day S2 pairing are computed.

var CFG = {
  start: '2020-01-01',
  endExclusive: '2026-01-01',
  years: [2020, 2021, 2022, 2023, 2024, 2025],
  maxL8PerYearSeason: 3,
  maxGapDays: 4,
  minL8ShoreClear: 0.50,
  minS2ShoreClear: 0.50,
  minJointShoreClear: 0.40,
  driveFolder: 'Fuxian_Q2_external_candidates',
  lakes: {
    Dianchi_Waihai: {
      title: 'Dianchi / Waihai main basin',
      // The northern Caohai compartment is not claimed as part of Waihai.
      seed: [102.7049, 24.8379],
      box: [102.57, 24.62, 102.87, 25.06],
      crs: 'EPSG:32648',
      minPriorKm2: 180, maxPriorKm2: 380
    },
    Erhai: {
      title: 'Erhai Lake',
      seed: [100.1960, 25.8000],
      box: [100.04, 25.56, 100.35, 26.02],
      crs: 'EPSG:32647',
      minPriorKm2: 150, maxPriorKm2: 350
    },
    Xingyun: {
      title: 'Xingyun Lake (near-domain diagnostic)',
      seed: [102.7820, 24.3370],
      box: [102.69, 24.25, 102.88, 24.43],
      crs: 'EPSG:32648',
      minPriorKm2: 15, maxPriorKm2: 80
    }
  }
};

var jrc2019 = ee.Image(
  ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
    .filter(ee.Filter.eq('year', 2019)).first()
).select('waterClass').gte(2);

function geoFor(key) {
  var spec = CFG.lakes[key];
  var box = ee.Geometry.Rectangle(spec.box, null, false);
  // Small rectangles avoid the Point-constructor error observed in some GEE
  // Code Editor sessions. This box lies inside the intended main water body.
  var seed = ee.Geometry.Rectangle([
    spec.seed[0] - 0.0001, spec.seed[1] - 0.0001,
    spec.seed[0] + 0.0001, spec.seed[1] + 0.0001
  ], null, false);
  var components = jrc2019.selfMask().toInt().reduceToVectors({
    geometry: box,
    scale: 30,
    geometryType: 'polygon',
    eightConnected: true,
    maxPixels: 1e8,
    tileScale: 4,
    geometryInNativeProjection: false
  });
  var matched = components.filterBounds(seed);
  var lake = ee.Feature(matched.first()).geometry().simplify(30);
  var roi = lake.buffer(1000, 30);
  var shore = lake.buffer(300, 30)
    .difference(lake.buffer(-300, 30), 30);
  var interior = lake.buffer(-300, 30);
  return {spec: spec, box: box, matched: matched, lake: lake,
    roi: roi, shore: shore, interior: interior};
}

function l8Valid(image) {
  return image.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(image.select('QA_RADSAT').eq(0))
    .and(image.select('SR_B2').mask())
    .rename('l8_valid').unmask(0).toByte();
}

function s2Valid(image) {
  var scl = image.select('SCL');
  return scl.neq(0).and(scl.neq(1)).and(scl.neq(3))
    .and(scl.lt(8)).and(image.select('B2').mask())
    .rename('s2_valid').unmask(0).toByte();
}

function fraction(mask, geometry, scale) {
  var value = mask.rename('quality').reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: geometry,
    scale: scale,
    maxPixels: 1e8,
    tileScale: 4
  }).get('quality');
  return ee.Number(ee.Algorithms.If(value, value, 0));
}

function s2OnDay(dayText, geometry) {
  var day = ee.Date(dayText);
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geometry)
    .filterDate(day, day.advance(1, 'day'));
}

function bestS2(l8Day, geometry) {
  var nearby = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geometry)
    .filterDate(l8Day.advance(-CFG.maxGapDays, 'day'),
      l8Day.advance(CFG.maxGapDays + 1, 'day'))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 85))
    .map(function(image) {
      var dateText = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');
      var gap = ee.Date(dateText).difference(l8Day, 'day').abs();
      var score = ee.Number(image.get('CLOUDY_PIXEL_PERCENTAGE'))
        .add(gap.multiply(7));
      return image.set({pair_date: dateText, pair_gap: gap,
        pair_score: score});
    }).sort('pair_score');
  var fallback = ee.Image.constant(0).rename('fallback').set({
    pair_date: '1970-01-01', pair_gap: 99, pair_score: 999
  });
  var chosen = ee.Image(ee.Algorithms.If(nearby.size().gt(0),
    nearby.first(), fallback));
  return {count: nearby.size(), image: chosen};
}

function candidateCollection(key, geo) {
  var all = ee.ImageCollection([]);
  CFG.years.forEach(function(year) {
    var annual = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
      .filterBounds(geo.lake)
      .filterDate(String(year) + '-01-01', String(year + 1) + '-01-01')
      .filter(ee.Filter.lt('CLOUD_COVER', 85));
    var dry = annual.filter(ee.Filter.or(
      ee.Filter.calendarRange(1, 4, 'month'),
      ee.Filter.calendarRange(11, 12, 'month')))
      .sort('CLOUD_COVER').limit(CFG.maxL8PerYearSeason)
      .map(function(image) { return image.set('scan_season', 'dry'); });
    var wet = annual.filter(ee.Filter.calendarRange(6, 10, 'month'))
      .sort('CLOUD_COVER').limit(CFG.maxL8PerYearSeason)
      .map(function(image) { return image.set('scan_season', 'wet'); });
    all = all.merge(dry).merge(wet);
  });
  var list = all.toList(all.size());
  return ee.FeatureCollection(list.map(function(element) {
    var image = ee.Image(element);
    var l8DayText = ee.Date(image.get('system:time_start'))
      .format('YYYY-MM-dd');
    var l8Day = ee.Date(l8DayText);
    var pair = bestS2(l8Day, geo.roi);
    var s2DayText = ee.String(pair.image.get('pair_date'));
    var s2Count = pair.count;
    // A zero-band collection must never reach Image.select/rename, even for a
    // rejected row. The constant zero image makes the max() band-safe.
    var s2Masks = s2OnDay(s2DayText, geo.roi).map(s2Valid)
      .merge(ee.ImageCollection([
        ee.Image.constant(0).rename('s2_valid').toByte()
      ]));
    var s2Mask = s2Masks.max().rename('s2_valid').unmask(0).toByte();
    var l8Mask = l8Valid(image);
    var l8Clear = fraction(l8Mask, geo.shore, 30);
    var s2Clear = fraction(s2Mask, geo.shore, 30);
    var jointClear = fraction(l8Mask.and(s2Mask), geo.shore, 30);
    var gap = ee.Number(pair.image.get('pair_gap'));
    var pass = ee.Number(l8Clear.gte(CFG.minL8ShoreClear))
      .multiply(ee.Number(s2Clear.gte(CFG.minS2ShoreClear)))
      .multiply(ee.Number(jointClear.gte(CFG.minJointShoreClear)))
      .multiply(ee.Number(s2Count.gt(0)));
    var score = jointClear.subtract(gap.multiply(0.02));
    return ee.Feature(null, {
      lake: key,
      season: image.get('scan_season'),
      l8_date: l8DayText,
      l8_system_id: image.get('system:id'),
      l8_cloud_scene_pct: image.get('CLOUD_COVER'),
      l8_clear_shore: l8Clear,
      s2_date: s2DayText,
      s2_scene_count: s2Count,
      s2_clear_shore: s2Clear,
      joint_clear_shore: jointClear,
      day_gap: gap,
      quality_score: score,
      pass_quality_num: pass,
      status: 'CANDIDATE_NOT_LOCKED'
    });
  }));
}

function buildCandidate(record, geo) {
  var source = ee.Image(record.l8_system_id);
  var l8Mask = l8Valid(source);
  var l8 = source.select(['SR_B2', 'SR_B3', 'SR_B4',
      'SR_B5', 'SR_B6', 'SR_B7'])
    .multiply(0.0000275).add(-0.2)
    .rename(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
    .addBands(l8Mask.rename('valid'))
    .toFloat().unmask(-9999).clip(geo.roi);
  var collection = s2OnDay(record.s2_date, geo.roi)
    .map(function(image) { return image.addBands(s2Valid(image)); });
  var s2Mask = collection.select('s2_valid').max()
    .unmask(0).rename('s2_valid').toByte();
  var s2 = collection.qualityMosaic('s2_valid')
    .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'SCL',
      's2_valid'])
    .toUint16().unmask(0).clip(geo.roi);
  var joint = l8Mask.and(s2Mask).rename('joint_valid');
  var masks = ee.Image.constant(0).byte()
    .paint(geo.interior, 1).rename('prior_in')
    .addBands(ee.Image.constant(0).byte()
      .paint(geo.shore, 1).rename('prior_shore'))
    .addBands(l8Mask)
    .addBands(s2Mask)
    .addBands(joint)
    .addBands(ee.Image.constant(0).byte()
      .paint(geo.roi, 1).rename('roi_zone'))
    .toByte().unmask(0).clip(geo.roi);
  return {l8: l8, s2: s2, masks: masks};
}

function exportCandidate(record, geo) {
  var products = buildCandidate(record, geo);
  var l8Tag = record.l8_date.replace(/-/g, '');
  var s2Tag = record.s2_date.replace(/-/g, '');
  var stem = 'Q2_CANDIDATE_' + record.lake + '_' + l8Tag;
  var region = geo.roi.bounds(30);
  // Zero-origin UTM grids make all dates at one lake exactly co-registered;
  // the 10 m Sentinel-2 grid nests in the 30 m Landsat grid.
  Export.image.toDrive({
    image: products.l8,
    description: stem + '_L8', folder: CFG.driveFolder,
    region: region, crs: geo.spec.crs,
    crsTransform: [30, 0, 0, 0, -30, 0], maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: -9999}
  });
  Export.image.toDrive({
    image: products.s2,
    description: stem + '_S2_' + s2Tag, folder: CFG.driveFolder,
    region: region, crs: geo.spec.crs,
    crsTransform: [10, 0, 0, 0, -10, 0], maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: 0}
  });
  Export.image.toDrive({
    image: products.masks,
    description: stem + '_masks', folder: CFG.driveFolder,
    region: region, crs: geo.spec.crs,
    crsTransform: [30, 0, 0, 0, -30, 0], maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true}
  });
}

var panel = ui.Panel({style: {width: '480px', padding: '10px'}});
panel.add(ui.Label('External-lake quality inventory',
  {fontSize: '18px', fontWeight: 'bold'}));
panel.add(ui.Label('Stage 1 only. Scan one lake, download its quality manifest, ' +
  'then inspect candidate imagery. No trained-model prediction is used.'));
var lakeKeys = ['Dianchi_Waihai', 'Erhai', 'Xingyun'];
var lakeSelect = ui.Select({items: lakeKeys, value: lakeKeys[0]});
panel.add(lakeSelect);
var status = ui.Label('Choose a lake and click Scan.');
panel.add(status);
var recordsPanel = ui.Panel();
panel.add(recordsPanel);
ui.root.insert(0, panel);

var prepared = {};
function scan() {
  recordsPanel.clear();
  var key = lakeSelect.getValue();
  var geo = geoFor(key);
  status.setValue('Checking the 2019 JRC main-lake polygon...');
  ee.Dictionary({count: geo.matched.size(),
    area_km2: geo.lake.area(30).divide(1e6)}).evaluate(function(result, error) {
    if (error) {
      status.setValue('STOP: prior geometry failed: ' + JSON.stringify(error));
      return;
    }
    if (result.count !== 1 ||
        result.area_km2 < geo.spec.minPriorKm2 ||
        result.area_km2 > geo.spec.maxPriorKm2) {
      status.setValue('STOP: check lake seed and 2019 polygon. ' +
        'Matched components=' + result.count +
        ', area=' + Number(result.area_km2).toFixed(1) + ' km2.');
      return;
    }
    status.setValue('Scanning ' + key + ' quality-only pairs; this may take a few minutes...');
    var all = candidateCollection(key, geo);
    all.evaluate(function(collection, scanError) {
      if (scanError) {
        status.setValue('Scan failed: ' + JSON.stringify(scanError));
        return;
      }
      var records = collection.features.map(function(feature) {
        return feature.properties;
      });
      var seen = {};
      records = records.filter(function(record) {
        if (seen[record.l8_date]) return false;
        seen[record.l8_date] = true;
        return true;
      });
      records.sort(function(a, b) {
        if (a.season !== b.season) return a.season < b.season ? -1 : 1;
        return Number(b.quality_score) - Number(a.quality_score);
      });
      var eligible = records.filter(function(record) {
        return Number(record.pass_quality_num) === 1 &&
          record.s2_date !== '1970-01-01';
      });
      var dryCount = eligible.filter(function(r) { return r.season === 'dry'; }).length;
      var wetCount = eligible.filter(function(r) { return r.season === 'wet'; }).length;
      status.setValue(key + ': JRC area ' + Number(result.area_km2).toFixed(1) +
        ' km2; ' + records.length + ' unique candidates; quality pass ' +
        dryCount + ' dry / ' + wetCount + ' wet. No date is locked yet.');
      if (records.length === 0) return;

      var exportRows = ee.FeatureCollection(records.map(function(record) {
        return ee.Feature(null, record);
      }));
      recordsPanel.add(ui.Button('Prepare this lake candidate CSV (1 task)',
        function() {
          Export.table.toDrive({collection: exportRows,
            description: 'Q2_CANDIDATE_' + key + '_quality_manifest',
            folder: CFG.driveFolder, fileFormat: 'CSV'});
          status.setValue('One candidate CSV task prepared. Start it in Tasks.');
        }));
      recordsPanel.add(ui.Label('Pass threshold: L8 and S2 shoreline clear >=50%; ' +
        'joint >=40%; date gap <=4 days. Failed rows remain in the CSV.'));
      function pickThree(season) {
        var pool = eligible.filter(function(r) { return r.season === season; });
        var picked = [];
        var usedYears = {};
        var usedS2 = {};
        pool.forEach(function(r) {
          var year = r.l8_date.slice(0, 4);
          if (picked.length < 3 && !usedYears[year] && !usedS2[r.s2_date]) {
            picked.push(r); usedYears[year] = true; usedS2[r.s2_date] = true;
          }
        });
        pool.forEach(function(r) {
          if (picked.length < 3 && picked.indexOf(r) < 0 &&
              !usedS2[r.s2_date]) {
            picked.push(r); usedS2[r.s2_date] = true;
          }
        });
        return picked;
      }
      var batch = pickThree('dry').concat(pickThree('wet'));
      if (batch.length === 6) {
        recordsPanel.add(ui.Button(
          'Prepare 3 dry + 3 wet: 18 GeoTIFF + 1 CSV tasks', function() {
            var fresh = batch.filter(function(record) {
              return !prepared[key + '_' + record.l8_date];
            });
            if (fresh.length !== 6) {
              status.setValue('Some batch dates were already prepared in this Run. ' +
                'Use the single-date button for the remaining dates.');
              return;
            }
            fresh.forEach(function(record) {
              prepared[key + '_' + record.l8_date] = true;
              exportCandidate(record, geo);
            });
            Export.table.toDrive({
              collection: ee.FeatureCollection(batch.map(function(record) {
                return ee.Feature(null, record);
              })),
              description: 'Q2_CANDIDATE_' + key + '_six_pair_manifest',
              folder: CFG.driveFolder, fileFormat: 'CSV'
            });
            status.setValue('18 candidate GeoTIFF + 1 manifest tasks prepared for ' + key +
              '. Start them in Tasks. Dates remain unsealed until review.');
            print('Q2_CANDIDATE_BATCH_' + key, batch);
          }));
        recordsPanel.add(ui.Label('Batch dates: ' + batch.map(function(r) {
          return r.l8_date + ' / ' + r.s2_date;
        }).join('; ')));
      } else {
        recordsPanel.add(ui.Label('Fewer than three distinct passing pairs ' +
          'in at least one season. Do not fill with unqualified scenes.'));
      }
      var choices = eligible.map(function(record) {
        return record.l8_date + ' [' + record.season + '] S2 ' + record.s2_date +
          ' | joint ' + (100 * record.joint_clear_shore).toFixed(1) + '%';
      });
      if (choices.length === 0) {
        recordsPanel.add(ui.Label('STOP: no pair passed the quality gate. ' +
          'Export only the CSV and report this result.'));
        return;
      }
      var select = ui.Select({items: choices, value: choices[0]});
      recordsPanel.add(select);
      var selected = function() {
        return eligible[choices.indexOf(select.getValue())];
      };
      recordsPanel.add(ui.Button('Show selected candidate', function() {
        var record = selected();
        var images = buildCandidate(record, geo);
        Map.layers().reset();
        Map.addLayer(images.l8, {bands: ['B4', 'B3', 'B2'],
          min: 0, max: 0.25}, 'L8 ' + record.l8_date, true);
        Map.addLayer(images.s2, {bands: ['B4', 'B3', 'B2'],
          min: 0, max: 3000}, 'S2 ' + record.s2_date, false);
        Map.addLayer(images.masks.select('joint_valid').not()
          .selfMask().clip(geo.shore), {palette: ['ff1744']},
          'Invalid common shoreline', false);
        Map.addLayer(ee.Image.constant(0).byte()
          .paint(geo.lake, 1, 1), {palette: ['ffff00']},
          '2019 JRC outline', false);
        Map.centerObject(geo.lake, 10);
        print('Q2_CANDIDATE_SELECTED', record);
      }));
      recordsPanel.add(ui.Button('Prepare selected candidate: 3 GeoTIFF tasks',
        function() {
          var record = selected();
          var id = key + '_' + record.l8_date;
          if (prepared[id]) {
            status.setValue('Tasks already prepared in this Run for ' + id + '.');
            return;
          }
          prepared[id] = true;
          exportCandidate(record, geo);
          status.setValue('Three candidate tasks prepared for ' + id +
            '. Start them in Tasks; these are not the locked test set.');
        }));
      recordsPanel.add(ui.Label('To inspect another lake, change the lake selector ' +
        'and click Scan again. Each lake needs its own candidate CSV task.'));
    });
  });
}
panel.insert(3, ui.Button('Scan selected lake', scan));
Map.setCenter(102.8, 24.6, 8);
