-- Starter passages for the Reading Aloud module.
--
-- Run once in the Supabase SQL editor, after schema.sql. Without this the
-- passages table is empty and the app silently falls back to a single
-- hard-coded line, so the pace score can never be exercised at more than one
-- level.
--
-- Re-running is safe: the insert is skipped entirely once any passage exists.
--
-- `level` picks the expected reading pace (see LEVEL_TARGET_WPM in
-- scoring-service/scoring.py): P1-P2 = 100 wpm, P3-P4 = 120, P5-P6 = 140.

insert into passages (text_hi, level, difficulty)
select v.text_hi, v.level, v.difficulty
from (
  values
    (
      'सूरज पूरब से निकलता है और पश्चिम में डूब जाता है। दिन में आसमान नीला दिखता है। रात को तारे चमकते हैं। चाँद हमें ठंडी रोशनी देता है।',
      'P1-P2', 1
    ),
    (
      'मेरा नाम आशा है। मैं दूसरी कक्षा में पढ़ती हूँ। मुझे कहानी की किताबें बहुत पसंद हैं। रोज़ शाम को मैं अपनी दादी के पास बैठती हूँ। दादी मुझे एक नई कहानी सुनाती हैं।',
      'P1-P2', 1
    ),
    (
      'हमारे गाँव के पास एक छोटी नदी बहती है। गर्मियों में हम वहाँ नहाने जाते हैं। नदी के किनारे बड़े-बड़े पेड़ खड़े रहते हैं। पेड़ों पर पक्षी अपने घोंसले बनाते हैं। शाम को सूरज ढलता है तो पानी सुनहरा दिखाई देता है।',
      'P3-P4', 2
    ),
    (
      'आज हमारे विद्यालय में खेल दिवस था। सुबह से ही मैदान में बहुत चहल-पहल थी। मैंने दौड़ में भाग लिया और दूसरा स्थान पाया। मेरे मित्र ने ऊँची कूद में पहला स्थान जीता। अध्यापक ने सबको पुरस्कार दिए और हमारा उत्साह बढ़ाया।',
      'P3-P4', 3
    ),
    (
      'भारत अनेक भाषाओं और संस्कृतियों का देश है। यहाँ हर राज्य की अपनी अलग पहचान है। त्योहारों के समय पूरा देश एक साथ खुशियाँ मनाता है। अलग-अलग बोलियाँ होने पर भी लोगों के मन आपस में जुड़े रहते हैं। यही विविधता हमारी सबसे बड़ी शक्ति है।',
      'P5-P6', 4
    ),
    (
      'पुस्तकालय ज्ञान का भंडार होता है। वहाँ शांति से बैठकर पढ़ने में विशेष आनंद आता है। हर अलमारी में अलग विषय की पुस्तकें सजी रहती हैं। एक अच्छी पुस्तक हमें नई दुनिया की सैर कराती है। इसीलिए पढ़ने की आदत जीवन भर काम आती है।',
      'P5-P6', 5
    )
) as v(text_hi, level, difficulty)
where not exists (select 1 from passages);
