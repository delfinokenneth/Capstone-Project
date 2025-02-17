-- phpMyAdmin SQL Dump
-- version 5.0.2
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Mar 18, 2022 at 12:54 PM
-- Server version: 10.4.14-MariaDB
-- PHP Version: 7.2.33

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `isent`
--

-- --------------------------------------------------------

--
-- Table structure for table `evaluation`
--

CREATE TABLE `evaluation` (
  `id` int(11) NOT NULL,
  `idteacher` int(11) NOT NULL,
  `idstudent` int(11) NOT NULL,
  `section1` varchar(1500) NOT NULL,
  `section2` varchar(1500) NOT NULL,
  `section3` varchar(1500) NOT NULL,
  `section4` varchar(1500) NOT NULL,
  `section5` varchar(1500) NOT NULL,
  `comment` varchar(1500) DEFAULT NULL,
  `pos` float DEFAULT NULL,
  `neu` float DEFAULT NULL,
  `neg` float DEFAULT NULL,
  `sentiment` varchar(250) DEFAULT NULL,
  `score` double DEFAULT NULL,
  `dateEvaluated` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `evaluation`
--

INSERT INTO `evaluation` (`id`, `idteacher`, `idstudent`, `section1`, `section2`, `section3`, `section4`, `section5`, `comment`, `pos`, `neu`, `neg`, `sentiment`, `score`, `dateEvaluated`) VALUES
(29, 18013672, 18013672, '5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', 'good teacher', 74.4, 25.6, 0, 'positive', 4.36, '2022-03-18 05:27:48'),
(35, 18013672, 18013672, '5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', 'dili kahibaw mutudlo', 0.43, 0.03, 0.53, 'negative', 0.74, '2022-03-18 06:30:44'),
(36, 18013672, 18013672, '5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', 'she is a bad teacher', 0, 46.2, 53.8, 'negative', 1.16, '2022-03-18 06:31:11'),
(37, 18013672, 18013672, '5,4,4,4,4,4,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', 'I don\'t really like her because she doesn\'t enter the class', 0.01, 0, 99.99, 'negative', 0, '2022-03-18 09:13:08'),
(38, 18013672, 18013672, '5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', 'She is a bad teacher', 0, 46.2, 53.8, 'negative', 1.16, '2022-03-18 06:32:38'),
(39, 18013672, 18013672, '5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', 'She is the worst teacher at the school because she is so strict', 0, 51.2, 48.8, 'negative', 1.28, '2022-03-18 06:33:05'),
(40, 18013672, 18013672, '5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,2', 'dili kahibaw mutudlo nya magsige ug absent', 0.26, 0.29, 99.45, 'negative', 0.02, '2022-03-18 06:43:08'),
(41, 18013672, 18013672, '5,4,4,4,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,3', 'ganahan kayko nimo  kay imo jud i consider ang situation sa student', 99.69, 0, 0.31, 'positive', 4.98, '2022-03-18 06:42:13'),
(42, 18013672, 18013672, '5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', 'Dili jud mi ma bored sa imo class sir haha', 99.94, 0.02, 0.04, 'positive', 5, '2022-03-18 06:38:27'),
(43, 18013672, 18013672, '5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', 'boring', 0, 0, 100, 'negative', 0, '2022-03-18 06:37:09'),
(44, 18013672, 18013672, '5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', 'not boring', 66.2, 33.8, 0, 'positive', 4.15, '2022-03-18 06:37:39'),
(45, 18013672, 18013672, '5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', 'She is always absent and not a good teacher', 57.56, 0, 42.44, 'positive', 2.88, '2022-03-18 07:39:12'),
(55, 18013672, 18013672, '5,4,5,4,5,4,5,4', '5,4,5,5,5,4,5,5,4,3,5,5,4,4,5,5', '5,4,5,3,2,4,5', '5,4,5,5,4,5,4,3,4,5,4', '5,4,5,4,5,4,3,3,4,5,4', 'irresponsible teacher', 0, 25.6, 74.4, 'negative', 0.64, '2022-03-18 06:49:44'),
(56, 18013672, 18013672, '5,4,5,5,5,4,5,4', '5,4,5,5,4,5,5,4,5,4,3,5,4,5,4,5', '5,4,5,5,5,3,2', '5,4,5,4,3,4,2,5,4,5,5', '5,4,5,5,5,5,5,5,4,5,4', 'happy teacher', 78.7, 21.3, 0, 'positive', 4.47, '2022-03-18 06:50:05'),
(57, 18013672, 18013672, '5,4,5,5,5,4,5,4', '3,4,3,2,4,5,4,5,5,4,5,5,4,5,5,3', '5,4,5,5,2,4,3', '5,4,5,4,3,4,5,5,5,5,4', '5,4,5,3,4,5,5,4,5,4,5', NULL, NULL, NULL, NULL, NULL, NULL, '2022-03-10 08:37:31'),
(58, 18013672, 18013672, '5,4,5,4,3,4,5,5', '5,4,5,4,3,4,5,5,4,5,4,3,4,5,4,3', '5,4,5,5,4,5,4', '5,4,5,5,4,5,4,3,5,4,5', '5,4,5,4,4,5,4,5,5,4,5', NULL, NULL, NULL, NULL, NULL, NULL, '2022-03-10 08:37:36'),
(59, 18013672, 18013672, '5,4,5,4,5,5,5,4', '5,4,5,4,5,4,5,4,5,5,4,5,4,4,5,4', '5,4,5,4,5,4,5', '5,4,5,5,4,5,4,5,4,5,4', '5,5,5,5,5,5,4,5,4,4,5', NULL, NULL, NULL, NULL, NULL, NULL, '2022-03-10 08:37:40'),
(60, 18013672, 18013672, '5,4,5,5,4,5,4,5', '5,4,5,4,5,5,4,5,5,4,5,5,4,5,5,4', '5,4,5,4,5,4,5', '5,4,5,4,5,5,4,5,4,5,5', '5,4,5,5,4,5,5,4,5,5,5', NULL, NULL, NULL, NULL, NULL, NULL, '2022-03-10 08:37:44'),
(61, 18013672, 18013672, '5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', NULL, NULL, NULL, NULL, NULL, NULL, '2022-03-10 08:37:47'),
(62, 18013672, 18013672, '5,4,5,5,5,5,5,5', '5,4,5,4,5,4,5,4,5,4,5,4,5,5,4,5', '5,4,4,5,5,4,5', '5,4,5,4,5,4,5,4,5,4,5', '5,4,5,4,5,5,5,4,5,4,5', NULL, NULL, NULL, NULL, NULL, NULL, '2022-03-10 08:37:50'),
(63, 18013672, 18013672, '5,4,5,4,5,5,4,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', NULL, NULL, NULL, NULL, NULL, NULL, '2022-03-10 08:37:53'),
(64, 18013672, 18013672, '5,4,5,5,5,5,5,5', '5,4,5,4,5,4,5,4,5,4,5,5,4,5,4,5', '5,4,5,4,5,5,4', '5,5,5,5,5,5,5,5,5,5,5', '5,4,5,4,5,4,5,4,3,4,5', NULL, NULL, NULL, NULL, NULL, NULL, '2022-03-10 08:37:56'),
(65, 18013672, 18013672, '5,4,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', 'testing comment yeah', 52.4, 47.6, 0, 'positive', 3.81, '2022-03-18 06:50:34'),
(66, 18013672, 18013672, '5,4,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', '5,5,5,5,5,5,5,5,5,5,5', NULL, NULL, NULL, NULL, NULL, NULL, '2022-03-10 08:37:59'),
(67, 18013672, 18013672, '5,4,5,5,4,5,5,4', '5,4,5,4,5,5,4,5,4,5,4,5,4,5,5,5', '5,4,5,4,5,4,5', '5,4,5,4,5,4,5,4,5,4,5', '5,4,5,5,4,5,4,5,4,5,5', NULL, NULL, NULL, NULL, NULL, NULL, '2022-03-10 08:38:02'),
(68, 18013672, 18013672, '5,4,5,5,4,5,4,5', '5,4,5,4,5,5,4,5,5,4,4,5,5,4,5,5', '5,4,5,4,5,4,5', '5,4,5,5,5,4,5,4,5,5,5', '5,4,5,5,4,5,5,5,5,4,5', NULL, NULL, NULL, NULL, NULL, NULL, '2022-03-10 08:44:22'),
(69, 18013672, 18013672, '1,1,1,1,1,1,1,1', '1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1', '1,1,1,1,1,1,1', '1,1,1,1,1,1,1,1,1,1,1', '1,1,1,1,1,1,1,1,1,1,1', NULL, NULL, NULL, NULL, NULL, NULL, '2022-03-10 08:47:03'),
(70, 18013672, 18013672, '4,4,4,4,4,4,4,4', '4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4', '4,4,4,4,4,4,4', '4,4,4,4,4,4,4,4,4,4,4', '4,4,4,4,4,4,4,4,4,4,4', NULL, NULL, NULL, NULL, NULL, NULL, '2022-03-10 08:56:08'),
(71, 18013672, 18013672, '5,4,5,5,4,5,5,4', '4,5,4,5,4,5,4,5,5,4,5,4,5,4,5,5', '4,5,4,5,5,4,5', '5,4,5,4,5,5,4,5,4,5,4', '5,4,5,4,5,4,5,5,4,5,5', 'test null comment', 0.02, 0.97, 0.01, 'neutral', NULL, '2022-03-18 09:42:01'),
(72, 18013672, 18013672, '5,4,5,4,5,4,5,4', '5,4,5,5,4,5,4,5,5,4,5,4,5,4,5,5', '5,4,5,4,5,4,5', '5,5,4,5,5,5,4,5,5,4,5', '5,4,5,4,5,4,5,5,4,5,4', 'so very responsible teacher', 48.9, 51.1, 0, 'positive', 3.72, '2022-03-18 04:41:09'),
(73, 18013672, 18013672, '5,4,5,4,5,4,5,5', '5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,5', '5,4,5,4,5,4,5', '5,4,5,4,5,5,4,5,5,4,4', '5,4,5,4,5,4,5,4,5,4,5', 'i am happy she is my teacher', 42.5, 57.5, 0, 'positive', 3.56, '2022-03-18 11:42:55');

-- --------------------------------------------------------

--
-- Table structure for table `questionaire`
--

CREATE TABLE `questionaire` (
  `id` int(11) NOT NULL,
  `section` int(11) NOT NULL,
  `question` text NOT NULL,
  `modifiedby` int(11) NOT NULL,
  `dateAdded` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `questionaire`
--

INSERT INTO `questionaire` (`id`, `section`, `question`, `modifiedby`, `dateAdded`) VALUES
(1, 1, 'uploads the learning materials to the LMS prior to the scheduled virtual or online class;', 0, '0000-00-00 00:00:00'),
(2, 1, 'provides a copy of the syllabus/course guide/lesson guide beforehand in order for us to prepare and study in advance', 0, '0000-00-00 00:00:00'),
(3, 1, 'provides timeline either on a separate document or in the embedded calendar section of the LMS to us at the start of the semester or on a monthly or weekly basis;', 0, '0000-00-00 00:00:00'),
(4, 1, 'presents the intended learning outcomes (ILO) before the start of the class;', 0, '0000-00-00 00:00:00'),
(5, 1, 'presents expectations and grading schemes and/or policies;', 0, '0000-00-00 00:00:00'),
(6, 1, 'facilitates in the review of the previews learning and relates it to the present lesson.', 0, '0000-00-00 00:00:00'),
(7, 1, 'utilizes online  course design that clearly articulates course policies and procedures;', 0, '0000-00-00 00:00:00'),
(8, 1, 'organizes course contents through modules, folders, drives or sections.', 0, '0000-00-00 00:00:00'),
(9, 2, 'always checks our attendance and punctuality;', 0, '0000-00-00 00:00:00'),
(10, 2, 'demonstrates appropriate depth of knowledge of the course;', 0, '0000-00-00 00:00:00'),
(11, 2, 'integrates current developments or research findings into the discussion;', 0, '0000-00-00 00:00:00'),
(12, 2, 'explains difficult terms, concepts or problems in more than one way;', 0, '0000-00-00 00:00:00'),
(13, 2, 'ensures that the teaching and learning activities are aligned with the intended learning outcomes;', 0, '0000-00-00 00:00:00'),
(14, 2, 'has well-modulated and audible voice, proper intonation, appropriate facial expression and body gestures as seen during live video conference/online class;', 0, '0000-00-00 00:00:00'),
(15, 2, 'makes the teaching and learning interesting that encourages us to participate;', 0, '0000-00-00 00:00:00'),
(16, 2, 'exposes us to varied teaching strategies which are level appropriate for us and for the online learning modality;', 0, '0000-00-00 00:00:00'),
(17, 2, 'utilizes appropriate online and offline applications and/or multimedia resources to enhance the onine teaching-learning process;', 0, '0000-00-00 00:00:00'),
(18, 2, 'adapts to the learning pace of the students accordingly;', 0, '0000-00-00 00:00:00'),
(19, 2, 'relates the topics and assignments to the course content and real-life situations;', 0, '0000-00-00 00:00:00'),
(20, 2, 'encourages us to interact with one another and with him/her;', 0, '0000-00-00 00:00:00'),
(21, 2, 'provides opportunities for us to engage in active learning (i.e peer review, interactive simulations, web search, experience-based projects and multimedia presentations);', 0, '0000-00-00 00:00:00'),
(22, 2, 'entertains openly to questions and gives proper responses;', 0, '0000-00-00 00:00:00'),
(23, 2, 'effectively handles inappropriate discussion postings or other unacceptable online behavior.', 0, '0000-00-00 00:00:00'),
(24, 2, 'organizes and explains online assignments and their corresponding due dates', 0, '0000-00-00 00:00:00'),
(25, 3, 'evaluates our understanding through assessment methods that are deemed appropriate for the online learning modality;', 0, '0000-00-00 00:00:00'),
(26, 3, 'returns corrected/rate outputs through the LMS or other online platforms/mediums when necessary;', 0, '0000-00-00 00:00:00'),
(27, 3, 'provides feedback to help us improve our works/outputs and performances;', 0, '0000-00-00 00:00:00'),
(28, 3, 'explains to us the rubrics or criteria to be used in rating or assignments, projects, performance tasks, recitations and other activities;', 0, '0000-00-00 00:00:00'),
(29, 3, 'give us formative type of assessment such as but not limited to discussion threads, online board discussion, blogs, informative chats and quizzes;', 0, '0000-00-00 00:00:00'),
(30, 3, 'gives us summative type of assessment such as but not limited to major exams, prohects, portfolios and presentations;', 0, '0000-00-00 00:00:00'),
(31, 3, 'keeps track of our progress through appropriate record-keeping systems or mechanisms.', 0, '0000-00-00 00:00:00'),
(32, 4, 'uses the LMS as our main online classroom/learning environment;', 0, '0000-00-00 00:00:00'),
(33, 4, 'provides navigational instructions in order to make the organization of the course easy to follow;', 0, '0000-00-00 00:00:00'),
(34, 4, 'provides orientation of the online course and its structure;', 0, '0000-00-00 00:00:00'),
(35, 4, 'makes sure that ambient/background noises are reduced to an acceptable level if not totally eliminated to avoid distractions during online classes;', 0, '0000-00-00 00:00:00'),
(36, 4, 'makes sure that there is sufficient lighting in order for us to see him/her clearly during onlince classes;', 0, '0000-00-00 00:00:00'),
(37, 4, 'makes sure that his/her background or surrounding environment during video conference is appropriate;', 0, '0000-00-00 00:00:00'),
(38, 4, 'uses a variety of online tools to facilitate our comprehension and engagement;', 0, '0000-00-00 00:00:00'),
(39, 4, 'exhibits creativity in presentong the topics during synchronous sessions;', 0, '0000-00-00 00:00:00'),
(40, 4, 'shows mastery in navigating the LMS, using its features and provides technical support when necessart especially during online class;', 0, '0000-00-00 00:00:00'),
(41, 4, 'responds to emails/chats/messages within 72 hours or less;', 0, '0000-00-00 00:00:00'),
(42, 4, 'creates a sense of community in the online course;', 0, '0000-00-00 00:00:00'),
(53, 5, 'starts and ends online class on time;', 0, '0000-00-00 00:00:00'),
(54, 5, 'attends online classes regularly;', 0, '0000-00-00 00:00:00'),
(55, 5, 'neat, well-groomed and dresses appropriately for online class;', 0, '0000-00-00 00:00:00'),
(56, 5, 'refrains from using foul and vulgar works both spoken and written at all times;', 0, '0000-00-00 00:00:00'),
(57, 5, 'shows respect care and concern for our welfare;', 0, '0000-00-00 00:00:00'),
(58, 5, 'shows passion, enthusiasm and excitement both in communicating with us and during online class;', 0, '0000-00-00 00:00:00'),
(59, 5, 'treats class members equitably and respectfully;', 0, '0000-00-00 00:00:00'),
(60, 5, 'responds constructively to our questions, opinions and other inputs.', 0, '0000-00-00 00:00:00'),
(61, 5, 'upholds confidentiality at all times and refrains from sharing our sensitive information (i.e grades, contact information, addess and etc.) to other members of the class and online;', 0, '0000-00-00 00:00:00'),
(62, 5, 'respects our privacy and observes class schedule or appropriate time for correspondence;', 0, '0000-00-00 00:00:00'),
(63, 5, 'encourages mutual respect among students.', 0, '0000-00-00 00:00:00');

-- --------------------------------------------------------

--
-- Table structure for table `section`
--

CREATE TABLE `section` (
  `id` int(11) NOT NULL,
  `section` int(11) NOT NULL,
  `name` varchar(50) NOT NULL,
  `description` varchar(50) NOT NULL,
  `percentage` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `section`
--

INSERT INTO `section` (`id`, `section`, `name`, `description`, `percentage`) VALUES
(1, 1, 'Preparation', 'The teacher...', 15),
(2, 2, 'Teaching and Learning Process', 'The teacher...', 35),
(3, 3, 'Assessment', 'The teacher...', 25),
(4, 4, 'Online Learning Environment', 'The teacher...', 10),
(5, 5, 'Teacher\'s Professionalism', 'The teacher...', 15);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `evaluation`
--
ALTER TABLE `evaluation`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `questionaire`
--
ALTER TABLE `questionaire`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `section`
--
ALTER TABLE `section`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `evaluation`
--
ALTER TABLE `evaluation`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=74;

--
-- AUTO_INCREMENT for table `questionaire`
--
ALTER TABLE `questionaire`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=64;

--
-- AUTO_INCREMENT for table `section`
--
ALTER TABLE `section`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
