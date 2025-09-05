#!/usr/bin/env python3
"""
종합 학습 시스템 테스트

전체 학습 파이프라인 검증:
1. 데이터 로드
2. 모델 학습 (기본/고급/하이브리드)
3. 관계성 학습
4. 템플릿 학습
5. 통합 예측
6. 성능 평가
"""

import json
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
from collections import defaultdict
import time

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent))

from config.settings import ApplicationConfig
from services.model_integration_service import ModelIntegrationService
from services.hybrid_ocr_labeler import HybridOCRLabeler
from services.relational_feature_extractor import EnhancedRelationalFeatureExtractor, RelationalModelIntegration
from services.template_matching_system import TemplateMatchingSystem


class ComprehensiveTrainingTest:
    """종합 학습 테스트 클래스"""
    
    def __init__(self):
        self.config = ApplicationConfig()
        self.logger = self._setup_logger()
        self.results = {
            'data_stats': {},
            'training_results': {},
            'evaluation_results': {},
            'errors': []
        }
        
    def _setup_logger(self):
        """로거 설정"""
        logger = logging.getLogger('comprehensive_test')
        logger.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    def load_training_data(self):
        """학습 데이터 로드"""
        self.logger.info("=" * 60)
        self.logger.info("1. 학습 데이터 로드")
        self.logger.info("=" * 60)
        
        # v2 라벨 데이터 로드
        labels_v2_dir = Path(self.config.processed_data_directory) / 'labels_v2'
        label_files = list(labels_v2_dir.glob('*.json'))
        
        self.logger.info(f"v2 라벨 파일 발견: {len(label_files)}개")
        
        # 이미지 파일 확인
        images_dir = Path(self.config.processed_data_directory) / 'images'
        image_files = list(images_dir.glob('*.png'))
        
        self.logger.info(f"이미지 파일 발견: {len(image_files)}개")
        
        # 데이터 로드
        training_data = []
        validation_data = []
        
        # 80/20 분할
        split_idx = int(len(label_files) * 0.8)
        
        for i, label_file in enumerate(label_files):
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 이미지 경로 확인
                base_name = label_file.stem.replace('_label_v2', '')
                image_path = images_dir / f"{base_name}.png"
                
                if image_path.exists():
                    data['image_path'] = str(image_path)
                
                if i < split_idx:
                    training_data.append(data)
                else:
                    validation_data.append(data)
                    
            except Exception as e:
                self.logger.error(f"파일 로드 실패 {label_file.name}: {e}")
                self.results['errors'].append(f"Data load error: {label_file.name}")
        
        self.logger.info(f"학습 데이터: {len(training_data)}개")
        self.logger.info(f"검증 데이터: {len(validation_data)}개")
        
        # 통계 수집
        self.results['data_stats'] = {
            'total_labels': len(label_files),
            'total_images': len(image_files),
            'training_samples': len(training_data),
            'validation_samples': len(validation_data)
        }
        
        # 문서 유형 분석
        doc_types = defaultdict(int)
        total_entities = 0
        
        for data in training_data:
            doc_type = data.get('document_metadata', {}).get('document_type', 'unknown')
            doc_types[doc_type] += 1
            total_entities += len(data.get('entities', []))
        
        self.logger.info(f"문서 유형: {dict(doc_types)}")
        self.logger.info(f"총 엔티티 수: {total_entities}")
        
        self.results['data_stats']['document_types'] = dict(doc_types)
        self.results['data_stats']['total_entities'] = total_entities
        
        return training_data, validation_data
    
    def test_hybrid_model(self, training_data, validation_data):
        """하이브리드 모델 테스트"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("2. 하이브리드 모델 (XGBoost/LightGBM/CRF) 학습")
        self.logger.info("=" * 60)
        
        try:
            # 하이브리드 모델 생성
            hybrid_model = HybridOCRLabeler(self.config, self.logger)
            hybrid_model.initialize()
            
            self.logger.info("하이브리드 모델 초기화 완료")
            
            # 학습
            start_time = time.time()
            self.logger.info(f"학습 시작 ({len(training_data)}개 문서)...")
            
            training_stats = hybrid_model.train(training_data)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"학습 완료 (소요 시간: {elapsed_time:.2f}초)")
            
            # 통계 출력
            self.logger.info(f"총 샘플: {training_stats.get('total_samples', 0)}")
            self.logger.info("모델별 성능:")
            for model, score in training_stats.get('model_performances', {}).items():
                self.logger.info(f"  - {model}: {score:.3f}")
            
            # 검증
            if validation_data:
                self.logger.info("\n검증 데이터로 평가 중...")
                eval_results = hybrid_model.evaluate(validation_data)
                
                accuracy = eval_results.get('accuracy', 0)
                self.logger.info(f"검증 정확도: {accuracy:.3f}")
                
                self.results['training_results']['hybrid_model'] = {
                    'training_time': elapsed_time,
                    'training_stats': training_stats,
                    'validation_accuracy': accuracy
                }
            
            return hybrid_model
            
        except Exception as e:
            self.logger.error(f"하이브리드 모델 학습 실패: {e}")
            self.results['errors'].append(f"Hybrid model error: {str(e)}")
            return None
    
    def test_relational_learning(self, training_data, hybrid_model):
        """관계성 학습 테스트"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("3. 관계성 특징 학습")
        self.logger.info("=" * 60)
        
        try:
            # 관계성 특징 추출기 생성
            rel_extractor = EnhancedRelationalFeatureExtractor()
            
            # 패턴 학습
            self.logger.info("관계 패턴 학습 중...")
            rel_extractor.learn_from_labeled_data(training_data[:10])  # 처음 10개만
            
            # 학습된 패턴 통계
            label_sequences = len(rel_extractor.learned_patterns.get('label_sequences', {}))
            group_templates = len(rel_extractor.learned_patterns.get('group_templates', {}))
            
            self.logger.info(f"학습된 라벨 시퀀스: {label_sequences}개")
            self.logger.info(f"학습된 그룹 템플릿: {group_templates}개")
            
            # 관계성 통합 테스트
            if hybrid_model:
                rel_integration = RelationalModelIntegration(hybrid_model)
                
                # 샘플 예측
                test_entities = training_data[0].get('entities', [])[:5]
                image_path = training_data[0].get('image_path')
                
                enhanced_predictions = rel_integration.enhance_predictions_with_relations(
                    test_entities, 
                    image_path
                )
                
                self.logger.info(f"관계성 강화 예측: {len(enhanced_predictions)}개")
                
                # 관계성 부스트 적용 개수
                boosted = sum(1 for p in enhanced_predictions if p.get('relation_boost', False))
                self.logger.info(f"관계성 부스트 적용: {boosted}개")
            
            self.results['training_results']['relational_learning'] = {
                'label_sequences': label_sequences,
                'group_templates': group_templates
            }
            
            return rel_extractor
            
        except Exception as e:
            self.logger.error(f"관계성 학습 실패: {e}")
            self.results['errors'].append(f"Relational learning error: {str(e)}")
            return None
    
    def test_template_learning(self, training_data):
        """템플릿 학습 테스트"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("4. 템플릿 매칭 시스템 학습")
        self.logger.info("=" * 60)
        
        try:
            # 템플릿 시스템 생성
            template_system = TemplateMatchingSystem()
            
            # 템플릿 학습
            self.logger.info("템플릿 학습 중...")
            templates = template_system.learn_template_from_documents(training_data[:20])  # 처음 20개
            
            self.logger.info(f"생성된 템플릿: {len(templates)}개")
            
            for template in templates:
                self.logger.info(f"  - {template.template_name}: {len(template.fields)}개 필드")
            
            # 템플릿 매칭 테스트
            if len(training_data) > 20:
                test_doc = training_data[20]
                matched_template, score = template_system.match_template(test_doc)
                
                if matched_template:
                    self.logger.info(f"\n매칭 테스트:")
                    self.logger.info(f"  매칭 템플릿: {matched_template.template_name}")
                    self.logger.info(f"  매칭 점수: {score:.3f}")
            
            # 통계
            stats = template_system.get_template_statistics()
            
            self.results['training_results']['template_learning'] = {
                'total_templates': stats['total_templates'],
                'document_types': dict(stats['document_types'])
            }
            
            return template_system
            
        except Exception as e:
            self.logger.error(f"템플릿 학습 실패: {e}")
            self.results['errors'].append(f"Template learning error: {str(e)}")
            return None
    
    def test_integrated_service(self, training_data):
        """통합 서비스 테스트"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("5. 모델 통합 서비스 테스트")
        self.logger.info("=" * 60)
        
        try:
            # 통합 서비스 생성
            integration_service = ModelIntegrationService(self.config, self.logger)
            
            # 초기화
            self.logger.info("서비스 초기화 중...")
            init_success = integration_service.initialize()
            
            if not init_success:
                self.logger.error("서비스 초기화 실패")
                return None
            
            self.logger.info("서비스 초기화 완료")
            
            # 모델 학습
            self.logger.info("\n전체 모델 학습 시작...")
            start_time = time.time()
            
            training_results = integration_service.train_models()
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"전체 학습 완료 (소요 시간: {elapsed_time:.2f}초)")
            
            # 결과 출력
            for model_name, result in training_results.items():
                if result:
                    self.logger.info(f"  - {model_name}: 학습 완료")
            
            # 예측 테스트
            if training_data:
                test_doc = training_data[0]
                test_ocr_results = [
                    {
                        'bbox': entity['bbox'],
                        'text': entity.get('text', {}).get('value', '')
                    }
                    for entity in test_doc.get('entities', [])[:5]
                ]
                
                self.logger.info("\n통합 예측 테스트...")
                prediction = integration_service.predict_labels('test.jpg', test_ocr_results)
                
                self.logger.info(f"사용된 모델: {prediction.get('model_used', 'unknown')}")
                self.logger.info(f"예측 신뢰도: {prediction.get('confidence', 0):.3f}")
                
                if prediction.get('model_details'):
                    self.logger.info("모델별 신뢰도:")
                    for model, conf in prediction['model_details'].items():
                        self.logger.info(f"  - {model}: {conf:.3f}")
            
            self.results['training_results']['integrated_service'] = {
                'initialization': init_success,
                'training_time': elapsed_time,
                'models_trained': list(training_results.keys())
            }
            
            return integration_service
            
        except Exception as e:
            self.logger.error(f"통합 서비스 테스트 실패: {e}")
            self.results['errors'].append(f"Integration service error: {str(e)}")
            return None
    
    def generate_report(self):
        """최종 보고서 생성"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("최종 보고서")
        self.logger.info("=" * 60)
        
        # 데이터 통계
        self.logger.info("\n[데이터 통계]")
        for key, value in self.results['data_stats'].items():
            self.logger.info(f"  {key}: {value}")
        
        # 학습 결과
        self.logger.info("\n[학습 결과]")
        for model_name, result in self.results['training_results'].items():
            self.logger.info(f"\n  {model_name}:")
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, float):
                        self.logger.info(f"    {key}: {value:.3f}")
                    else:
                        self.logger.info(f"    {key}: {value}")
        
        # 오류
        if self.results['errors']:
            self.logger.info("\n[오류 발생]")
            for error in self.results['errors']:
                self.logger.info(f"  - {error}")
        else:
            self.logger.info("\n[오류 없음] ✓")
        
        # 시스템 상태 평가
        self.logger.info("\n[시스템 상태 평가]")
        
        total_tests = 5
        passed_tests = len(self.results['training_results'])
        success_rate = (passed_tests / total_tests) * 100
        
        self.logger.info(f"  테스트 통과: {passed_tests}/{total_tests} ({success_rate:.0f}%)")
        
        if success_rate >= 80:
            self.logger.info("  상태: ✅ 우수 - 프로덕션 준비 완료")
        elif success_rate >= 60:
            self.logger.info("  상태: ⚠️ 양호 - 일부 개선 필요")
        else:
            self.logger.info("  상태: ❌ 개선 필요 - 추가 작업 필요")
        
        # 보고서 저장
        report_path = Path(self.config.processed_data_directory) / 'reports' / f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"\n보고서 저장: {report_path}")
        
        return self.results
    
    def run_comprehensive_test(self):
        """종합 테스트 실행"""
        self.logger.info("=" * 60)
        self.logger.info("YOKOGAWA OCR 종합 학습 시스템 테스트")
        self.logger.info("=" * 60)
        self.logger.info(f"시작 시간: {datetime.now()}")
        
        try:
            # 1. 데이터 로드
            training_data, validation_data = self.load_training_data()
            
            if not training_data:
                self.logger.error("학습 데이터가 없습니다")
                return self.results
            
            # 2. 하이브리드 모델 테스트
            hybrid_model = self.test_hybrid_model(training_data, validation_data)
            
            # 3. 관계성 학습 테스트
            rel_extractor = self.test_relational_learning(training_data, hybrid_model)
            
            # 4. 템플릿 학습 테스트
            template_system = self.test_template_learning(training_data)
            
            # 5. 통합 서비스 테스트
            integration_service = self.test_integrated_service(training_data)
            
        except Exception as e:
            self.logger.error(f"종합 테스트 중 오류 발생: {e}")
            self.results['errors'].append(f"Comprehensive test error: {str(e)}")
        
        finally:
            # 최종 보고서 생성
            self.generate_report()
            
            self.logger.info(f"\n종료 시간: {datetime.now()}")
            self.logger.info("=" * 60)
            self.logger.info("테스트 완료")
            self.logger.info("=" * 60)
        
        return self.results


def main():
    """메인 실행 함수"""
    # 테스트 실행
    tester = ComprehensiveTrainingTest()
    results = tester.run_comprehensive_test()
    
    # 간단한 성공/실패 판정
    if results['errors']:
        print(f"\n⚠️ 테스트 완료 - {len(results['errors'])}개 오류 발생")
        sys.exit(1)
    else:
        print("\n✅ 모든 테스트 성공적으로 완료!")
        sys.exit(0)


if __name__ == "__main__":
    main()