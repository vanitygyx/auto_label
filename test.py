# -*- encoding:utf-8 -*-
from flask import Flask, request, jsonify
from tools.to_test import NER_text_process
from run_ner_crf_new import ner_test
from utils.to_test_fine_grad import RE_text_process
from utils.auto_extraction import re_test

text ="-----------------------------------------------------------------------请手动去除页眉页脚----------------------------------------------------------------------------\n\n第4期2022年4月 Journal \nof \nCAEITVol.17 \nNo.4Apr.2022　􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀤋􀦋 􀦋􀦋 􀦋电子信息领域发展综述 doi: \n10.3969\/j.issn.1673-5692.2022.04.003\n收稿日期:2022-03-10　　修订日期:2022-03-29基金项目:国防科技战略先导计划2021年外军情报侦察领域发展综述郭敏洁(中国电子科技集团公司第十研究所,四川 \n成都　610036)\n摘　要:2021年,美国再提“侦察威慑”概念,升级 \n“太平洋威慑倡议”,在印太地区的作战能力投资增加。面对复杂多样的世界环境,美国聚力“大国竞争”,将天军纳入情报界,设立首席数据和人工智能官推进以数据为中心的战略,利用人工智能等新技术优势提升安全竞争时代的“发现”能力。\n纵观本年度世界大国的情监侦领域发展态势,以美国为首的军事强国面向联合全域作战需求,着力提升跨域协同的情报获取能力,利用开放体系架构技术、人工智能技术、云技术等赋能情报处理体系,加速推进情监侦装备发展和演习实验,力图为联合全域作战提供感知优势。\n关键词: \n侦察威慑;大国竞争;情监侦;联合全域作战中图分类号:E712;TN915. \n0　　文献标志码:A　　文章编号:1673-5692(2022)04-324-05\nComprehensive \nAnalysis \nof \nAnnual \nDevelopment \nof \nthe \nIntelligence \nsu \nSurveillance \nand \nReconnaissance \nin \n2021\nGUO \nMin-jie(The \n10th \nResearch \nInstitute \nof \nCETC,Chengdu \n610036,China)\nAbstract: \nIn \n2021, \nthe \nUnited \nStates \nput \nforward \nthe \nconcept \nof \n“Deterrence \nby \nDetection” \nagain, \nup-grade \nthe \n“Pacific \nDeterrence \nInitiative”, \nand \nincrease \ninvestment \nin \ncombat \ncapabilities \nin \nthe \nIndo-Pa-cific \nregion. \nIn \nthe \nface \nof \na \ncomplex \nand \ndiverse \nworld \nenvironment, \nthe \nUnited \nStates \nis \nfocusing \non \n“great \npower \ncompetition”, \nincorporating \nthe \nSpace \nForce \ninto \nthe \nIntelligence \nCommunity, \nsetting \nup \na \nchief \ndata \nand \nartificial \nintelligence \nofficer \nto \npromote \nthe \ndata-centric \nstrategy, \nand \nusing \nartificial \nintelli-gence \nand \nother \nnew \ntechnological \nadvantages \nto \nenhance \n“discovery” \ncapabilities \nin \nthe \nera \nof \nsecurity \ncompetition. \nThroughout \nthe \ndevelopment \ntrend \nof \nthe \nworld’s \nmajor \npowers \nin \nthe \nfield \nof \nthe \nISR \nthis \nyear, \nthe \nmilitary \npowers \nheaded \nby \nthe \nUnited \nStates \nare \nfacing \nthe \nneeds \nof \njoint \nall-domain \noperations, \nfocusing \non \nimproving \nthe \nintelligence \nacquisition \ncapability \nof \ncross-domain \ncoordination, \nand \nusing \nopen \narchitecture \ntechnology, \nartificial \nintelligence \ntechnology, \ncloud \ntechnology \nand \nother \ntechnologies \nto \nem-power \nintelligence \nprocessing \nsystem, \naccelerate \nthe \ndevelopment \nof \nISR \nequipment \nand \nexercise \nexperi-ments, \nand \nstrive \nto \nprovide \nperception \nadvantages \nfor \njoint \nall-domain \noperations.\nKey \nwords: \ndeterrence \nby \ndetection;great \npower \ncompetition;ISR;joint \nall-domain \noperations0　引　言2021年,全球多个地区动荡不定,一系列国际大事紧锣密鼓的发生,同时,“安全竞争”“极限竞争”“稳定竞争”等新名词不断出现,表明国际体系竞争愈发激烈,大国角力愈发突出,情报侦察也越来越受到各国的重视。总体而言,2021年情报侦察领\n\n-----------------------------------------------------------------------请手动去除页眉页脚----------------------------------------------------------------------------\n\n2022年第4期 郭敏洁:2021年外军情报侦察领域发展综述 325　　\n　域呈现以下四个发展态势。\n1　安全竞争时代的“威慑”对抗指引\n情监侦发展方向　　随着大国竞争形式逐渐转向“安全竞争”,即不再以追求“你死我活”的零和博弈为宗旨,代之以压倒对方斗争意志而迫使对方屈服为目的,“威慑”思想全方面地指引着情监侦发展方向。美国《2022财年国防授权法案》批准71亿美元用于加强印太地区的军事态势,明确针对情监侦领域提出了多方面的授权支持,不仅重点关注军种现代化和情监侦能力的发展,还阐述了数据战略、军事计划、情报机构设置等内容,为美在大国竞争下的情报备战指明了方向。同时,多个智库报告也强调安全竞争时代的“威慑”对抗,这与官方思想相呼应。7月,美智库战略与预算评估中心(CSBA)发布《实施侦察威慑:在印太地区提高态势感知的创新能力、程序和组织》报告,进一步扩展与深化2020年提出的“侦察威慑”概念,评估了如何利用现有平台和新兴能力来提高印太地区的态势感知能力。8月,美国国际战略研究中心(CSIS)发布《发展现代化的情监侦,提升安全竞争时代的“发现”能力》评论文章聚焦情监侦领域,提出了美国及其伙伴亟需通过一系列措施完善现有的情监侦体系,以谋求安全竞争时代的信息优势。\n美国陆军率先从情监侦角度出台了响应大国竞争时代的军种顶层规划,于1月正式公开《美国陆军未来司令部情报概念2028》,该文件为美陆军未来情报发展奠定了理论基础,将对美陆军获取情报跨域优势产生深远影响。\n智库层面,2021年相继出台了多份报告,探讨使用人工智能(AI)等新兴技术来应对未来战争中的情监侦威胁。包括:CSIS于1月发布《保持情报优势:通过创新重塑情报》,指出情报界需使用AI来应对未来全球威胁;兰德公司于1月发布《美空军情报分析的技术创新与未来》,探讨AI如何在高压环境下将正确的情报传递给正确的人员;米特公司于3月发布《未来的情报:确保在未来战场上的决策优势———高超声速作战节奏下的情报》,研究利用AI技术重塑战场情报工作以确保决策优势。\n2　基于体系化感知目标,发展联合全\n域态势感知装备　　未来冲突将越来越呈现出跨域、多域和多职能的性质,要实现从“军种联合”向“跨域协同”,再向“多域融合”的深层次发展,不仅需要从组织上作出改变,更需要大量新型装备和技术,对情报侦察也提出了新的需求。\n2021年,美军在传感器技术标准领域取得了快\n速发展,为加强体系跨域作战能力,实现平台间的互联互通奠定了基础。开放组织下属的SOSA联合会于9月27日正式发布了首个开放式体系架构军用传感器标准———SOSATM参考体系架构技术标准1.0版,用于支持美国国防部指挥、控制、通信、计算机、网络、情报、监视和侦察(C5ISR)的开发。\n(1)各国加紧建设反高超声速目标探测系统高超音速武器是重要的非核战略威慑手段,是实现“全球快速打击”的重要抓手。同时,因其可在临近空间高速、高机动性飞行,难以被传统的侦察手段发现,针对新时期面临的空天威胁,各国加紧建设高超声速目标预警探测系统。美国开始建造能够跟踪高超音速武器的原型卫星,美国导弹防御局(MDA)在1月分别授予L3公司和诺格公司合同,为“高超音速和弹道跟踪太空传感器”(HBTSS)项目建造中视场(MFOV)原型卫星,标志着新威胁目标探测技术从概念设计迈向原型建造新阶段。11月,由诺格公司设计的HBTSS通过了一项关键设计评审,意味着MDA认可了诺格公司从低地球轨道探测和跟踪弹道和高超导弹的技术方法。同时,日本在2021财年预算中划拨1.7亿日元用于研发可跟踪高超声速武器的卫星星座。\n(2)全球商用天基情监侦能力迅猛发展以美军为首的军事强国正在持续加大力度将快速发展的新型商用卫星能力引入到其空间情报体系中,构建“国家+商用的综合天基情监侦体系”。\n在图像卫星上,美国国家侦察局延长并扩展了与行星联邦公司的商业卫星图像合同,根据合同,行星联邦公司将继续每日向美国情报和国防部门提供3 \nm~5 \nm分辨率的非机密图像,同时提供有限的视频图像服务。\n在射频卫星上,鹰眼360公司于2月推出用于射频地理空间情报分析的商业平台“太空任务”,通过了可视化全球射频活动的整体图景,并在6月成功发射3颗“集群3”(Cluster3)小卫星,显著扩大星座的全球重访和数据收集能力,鹰眼360公司还在2021年和2022年启动7颗额外的下一代集群,组成基线星座,将重放率降低到20 \nmin,为时敏防务、安全和商业应用提供支持,加速利用射频地理空间情\n\n-----------------------------------------------------------------------请手动去除页眉页脚----------------------------------------------------------------------------\n\n326　　\n 2022年第4期　报对抗对手国家。此外,美国Spire公司计划利用气象纳米卫星获取天基信号情报,这是Spire公司气象纳米卫星的新用途,美国政府、军方及情报界用户以及英国政府都对此极为关注。\n在合成孔径雷达卫星上,初创企业加快了SAR卫星星座的建设。6月30日,美国本影公司发射了首颗商用 \nSAR卫星“本影-2001”,配备了X波段的合成孔径雷达,可在16 \nkm2的区域内以0.25 \nm的分辨率获取图像。卡佩拉公司又相继发射了4颗卫星,随后又赢得美太空发展局导弹探测跟踪的遥感技术研究合同,并正式推出其开放数据计划,提供对基础SAR数据的访问。芬兰冰眼公司发射了4颗新型SAR卫星,配备了该公司最新的SAR卫星技术,将实现创新型SAR成像能力。美陆军寻求与冰眼公司合作,将SAR技术应用于陆军任务中,以缩短杀伤链闭合时间。日本SAR卫星数据和分析解决方案提供商Synspective公司宣布成功地从自己的首颗SAR卫星“Strix-α”中获取了第一张图像,这是日本首次成功从太空获取商业SAR卫星(100公斤级)图像。\n(3) \n军事强国加强空间侦察能力美国空间侦察相关机构上做出重大调整,不仅将美国天军正式加入美国情报界,成为情报界的第18名成员,还明确了太空职能单位的势力范围,通过签署保密的《受保护的国防战略框架》协议,明确了美国家地理空间情报局、美太空军和美国太空司令部的各自职责,消除各相关机构间长期存在的利益冲突,推动空间侦察相关机构开展“前所未有的”紧密协作。\n同时,多个军事强国也加强了传统侦察卫星的更新换代。俄罗斯发射了“北极”水文气象和气候监测系统的首颗气象卫星“北极-M”;韩国从2022年起开发一种基于微卫星的侦察系统,以增强其探测朝鲜机动导弹发射器等安全威胁的能力;法国3颗“天基信号情报能力”卫星成功发射升空,将构成法国首个信号情报卫星系统,使法国更好地收集太空电磁信号,为法国提供下一代空间监视能力。\n(4) \n积极研发并升级有人\/无人侦察装备为适应未来作战,美军积极研发与升级有人与无人侦察装备,并从机构建设、演示验证、项目技术等多角度推进有人\/无人协同侦察作战,旨在大幅度提升“单向透明”的态势感知优势。\n有人侦察系统普遍向高性能、综合化发展。美陆军的新侦察机项目———未来攻击侦察机正在有序推进,已于3月进入项目原型机竞争演示工作阶段。\n4月,以色列空军新型“奥龙”情报收集飞机进驻以\n色列内瓦蒂姆空军基地,可收集电子情报\/通信情报传感器数据。8 \n月,美陆军首次成功试飞新型机载侦察和电子战系统飞机,该飞机将帮助美陆军实现机载情监侦能力现代化,未来还将成为“高精度探测和开发系统”计划的一部分。12月,美国空军研究实验室在网上公开了关于Mayhem \n高超声速飞行器项目的合同文件,该系统可搭载响应 \nISR载荷,未来将用于侦察\/打击任务。\n无人侦察系统多样化发展,支持陆海空多域战场情报获取。无人侦察机方面,美海军的两架MQ-4C“海神之子”完成首次日本轮换部署任务,将为美国第7舰队提供海上监视和持续情监侦能力;美军的复仇者无人机演示了装备支持静默攻击的“军团”光电吊舱,进行了自动跟踪目标测试;以色列推出执行海上情报监视任务的“轨道器-4”无人机;乌克兰新型情报搜集无人机开始进行飞行测试。无人侦察车方面,韩国新型6×6无人侦察车研制成功,美国一公司推出了专为美海军陆战队的高级侦察车计划而制造的“水腹蛇”无人侦察车。海上无人侦察方面,美军升级了MK-18 \nMod \n2无人潜航器的传\n感器,接收了“海鹰”号无人水面舰艇,此外法国也研制出一种能为舰队安全航行提供环境信息的低成本水下滑翔器,英国海军测试用于调查未知水域信息的“水獭”无人勘测船。\n有人\/无人协同侦察将成为未来作战的常态。\n在海域方面,美国海军成立第59特遣部队,旨在集成新型和具有潜力的无人、人工智能赋能系统,用于增强海域感知,提高威慑力,并开展“无人系统综合作战问题21”演习,此次演习聚焦无人侦察系统与现有有人侦察装备体系的协同作战。在空域方面,美空军研究实验室宣布已成功进行了XQ-58A“女武神”无人机空射“空射管内集成无人系统-600”(ALTIUS,“阿尔提乌斯”)的试验,标志着美军完成了从有人机-无人机、直升机-无人机向无人机-无人机的协同侦察作战模式的突破。\n3　数据驱动的新质作战能力正凸显其\n作战效能　　数据已成为战场制信息权的核心驱动力,以“数据为中心”的新质作战能力正凸显其极高的作战效能。在情监侦领域,数据作为战略资产可为传\n\n-----------------------------------------------------------------------请手动去除页眉页脚----------------------------------------------------------------------------\n\n2022年第4期 郭敏洁:2021年外军情报侦察领域发展综述 327　　\n　感器数据管理、情报数据处理提供有利条件。为此,美国正在持续通过战略规划、项目布局等方式提升数据治理水平。\n2021年3月3日,在FCW数字政府峰会上,美\n国防部首席数据官戴夫·斯皮克表示,国防部去年10月发布的《数据战略》的3大重点领域为:1)联合全域作战,利用数据取得战场优势;2)高级领导决策支持,利用数据提升国防部的管理;3)业务分析,利用数据驱动各个层级的洞察和决策。5月5日,美国防部发布《创造数据优势》备忘录,以将国防部转型为以数据为中心的组织,目的是“提高战斗性能,并在从作战空间到理事会会议的所有层级创造决策优势”。 \n6月24日,美海军代理部长签署数据\n优势备忘录,确认海军部支持国防部副部长的数据战略,并指示海军部必须采取行动,以实现数据战略目标以及利用数据获得决策优势的愿景。10月,美国防部国家地理空间情报局发布《数据战略2021:当前与未来的任务》,提出“快速、精准、安全地创建、管理和分享可信数据”的愿景,旨在为处理大量地理空间情报数据提供有效、便捷的途径,从而为美情报界、军方及相关决策者提供有价值的情报。12月,美国防部正式设立首席数据和人工智能官(CD-AO),负责监督多个先前存在的办公室,包括首席数据官办公室(CDO)、联合人工智能中心(JAIC)和国防数字服务办公室(DDS)。\n3月23日, \nBluestaq公司赢得美太空军空间态势感知数据库———统一数据库(UDL)合同,该数据库旨在收集和整合来自军事、情报界、商业和外国的空间目标跟踪数据,其最初目标是为数据提供单一位置,并帮助简化数据权限管理。3月31日,美国防情报局(DIA)发布了“机器辅助分析快速存储数据库系统”(MARS)的第二个最小化可行性产品,用于提供初始作战序列能力。该产品将基于对手军队部署的地理位置及其配属装备,推理判断出该部队的层级与兵力结构。美国防情报局发布“机器辅助分析快速存储系统数据库”(MARS)的第二个最低化可行性产品,用于提供初始战斗序列能力。\n在技术应用方面,诺格公司计划将Deepwave数字公司的人工智能解决方案集成到目前或近期存在的一些机载和太空有效载荷上,将数据处理推向了更接近收集点的位置,让有效载荷自行处理数据,减少需要传输的数据量,实现情报产品的更快交付。\n美国陆军工程师和信息技术专家正致力于将“造雨者”(Rainmaker)数据结构程序集成至该军种“综合战术网络”(ITN)内的关键应用中,陆军C5ISR中心正在与该军种的远程精确火力跨职能小组(LRPF \nCFT)协调,以通过“造雨者”和其他数据结构系统缩短传感器到射手的数据传输时间,未来几年还将利用新型数据架构Data \nFabric向武器系统提供关键信息。\n4　情监侦新技术不断取得重大突破\n2021年,量子传感、数据处理、光学成像、认知\n对抗等多个技术取得重要突破,将对情报侦察领域产生深远影响。\n美陆军研究实验室利用Rydberg量子接收机首次探测了现实世界真实全频谱无线电信号,包括调幅(AM)、调频(FM)、Wi-Fi、蓝牙信号以及其他信号。DARPA启动了“量子孔径”(QA)项目,通过采用量子传感技术,开发一种全新的射频天线,可对射频信号侦察定位的新概念或方法带来颠覆性的影响。\n在数据处理领域,DARPA启动像素智能处理(IP2)项目,将人工智能引入边缘高端视频处理的嵌入式计算领域。同时,2021年,美国在DNA数据存储技术领域取得一批实用性成果,DNA数据存储联盟发布了首份白皮书———《保护我们的数字遗产:DNA数据存储简介》,规划技术发展方向,DNA存储在编码、合成、存储和检索等方面接连取得突破性发展,有望解决技术层面上面临的成本、效率和不稳定问题,将极大地缩短DNA存储技术迈向实用化的进程,未来可应用在海量数据存储、机密数据存储与传递等方面,具有巨大军事应用前景。\nDARPA的极限光学和成像(EXTREME)取得阶段性成果,展示了更小、更轻、功能更强大的透镜材料,能够实现传统光学系统尺寸、重量和功率 \n(SWaP)特性的革命性改进,将有效解决微小型无人机载荷能力不足的问题,为在竞争空域中飞行侦察的无人蜂群赋能。这种具有新颖光学特性的材料正在为政府和军事成像系统提供新的功能,已在集成紧凑型光电\/红外系统 \n(ICES)、XQ-58实验性隐形无人作战飞行器和空射无人机(ALOBO)等颠覆性能力项目中应用。\n此外,2021年美军通过多项举措关注对虚假信息的识别和分析能力。DARPA推进“语义取证”(SemaFor)、“不同来源主动诠释”(AIDA)项目;兰德公司发布《建立基于人工智能的反虚假信息框\n\n-----------------------------------------------------------------------请手动去除页眉页脚----------------------------------------------------------------------------\n\n328　　\n 2022年第4期　架》;美国会研究服务处发布更新版《“深度造假”与国家安全》报告。\n5　结　语\n未来的竞争环境必然呈现“高强度对抗”特性,作战部队将在分散、不连续、间歇和有限的情报环境下进行复杂多域、多安全域的连续作战。单一功能的情报侦察系统装备的生存力和作战性能面临严峻考验,采用分布式协同概念和人工智能等一系列新型技术,融合战场空间所有作战平台感知能力及感知数据的联合全域智能感知情报侦察体系才能满足未来战争对近实时、高精度、可执行情报的需求,支撑对敌方整个作战体系的感知识别。\n参考文献:[1]　Fiscal \nYear \n2022 \nNational \nDefense \nAuthorization \nAct[EB\/OL]. \n(2021-7-23)[2021-12-29]. \nhttps:\/\/www.\narmed-services.senate.gov\/imo\/media\/doc\/FY22%20 \nNDAA%20Executive%20Summary.pdf.\n[2]　MAHNKEN \nT \nG, \nSHARP \nT, \nBASSLER \nC, \net \nal. \nImple-menting \nDeterrence \nby \nDetection: \nInnovative \nCapabili-ties, \nProcesses, \nand \nOrganizations \nfor \nSituational \nAware-ness \nin \nthe \nIndo-Pacific \nRegion[EB\/OL]. \n(2021-7-14)\n[2021-12-29]. \nhttps:\/\/csbaonline.org\/uploads\/docu-ments\/CSBA8269_(Implementing_Deterrence_By_Detec-tion)_FINAL_web.pdf..\n[3]　HARRINGTON \nJ, \nMCCABE \nR. \nModernizing \nIntelli-gence, \nSurveillance, \nand \nReconnaissance \nto \n‘Find’ \nin \nthe \nEra \nof \nSecurity \nCompetition[EB\/OL]. \n(2021-8-6)\n[2021-12-29]. \nhttps:\/\/www.csis.org\/analysis\/modern-izing-intelligence-surveillance-and-reconnaissance-find-era-security-competition.\n[4]　AFC \nPam \n71-20-3: \nArmy \nFutures \nCommand \nConcept \nfor \nIntelligence \n2028[EB\/OL].(2021-1-5)[2021-12-29].  \nhttps:\/\/api.army.mil\/e2\/c\/downloads\/2021\/01\/05\/26b729a6\/20200918-afc-pam-71-20-3-intelligence-con-cept-final.pdf, \n2021-1-5.\n[5]　The \nOpen \nGroup \nSOSA \nConsortium. \nAFLCMC-2021-0153,Technical \nStandard \nfor \nSOSATM \nReference \nArchitec-ture,Edition \n1.0 \n[S]. \nSan \nFrancisco: \nThe \nOpen \nGroup \nSOSA \nConsortium, \n2021.\n[6]　COLIN \nCLARK. \nNRO, \nNGA, \nSPACECOM, \nSpace \nForce \nHammer \nOut \nBoundaries[EB\/OL]. \n(2021-8-24) \n[2021-12-29].https:\/\/breakingdefense.com\/2021\/08\/nro-nga-spacecom-space-force-hammer-out-boundaries\/.\n[7]　Ceres \nReconnaissance \nSpace \nSystem \nDesigned \nBy \nAirbus \nAnd \nThales \nSuccessfully \nLaunched[EB\/OL]. \n(2021-11-16)[2021-12-29].https:\/\/www.thalesgroup.com\/en\/group\/press_release\/ceres-reconnaissance-space-system-designed-airbus-and-thales-successfully, \n2021-11-16.\n[8]　SEFFERS \nG \nI. \nDIA \nPoised \nto \nRelease \nNext \nMARS \nPro-gram \nProduct[EB\/OL].(2021-3-30) \n[2021-12-29]. \nhttps:\/\/www.afcea.org\/content\/dia-poised-release-next-mars-program-product#:~:text=The%20new%20module%2C%20known%20as%20Order%20of%20Battle%2C,up%20intelligence%20analysts%20to%20perform%20more%20complex%20analysis, \n2021-\n3-30. \n[9]　李海龙,杨宏亮,朱方杰,等.加快基于网络信息体系侦察情报能力建设的几点思考[J].中国电子科学研究院学报,2019,14(4):338-341.\n作者简介郭敏洁(1989—),硕士,主要研究方向为科技情报研究。\n\n-----------------------------------------------------------------------请手动去除页眉页脚----------------------------------------------------------------------------\n\n"
sub_splite_text,split_text= NER_text_process(text)
entities = ner_test(sub_splite_text,split_text)
print(entities)
print(text[144:148])